import type {
  ExtensionAPI,
  ExtensionCommandContext,
  ExtensionContext,
  ProviderModelConfig,
  Theme,
} from "@earendil-works/pi-coding-agent";
import { matchesKey, visibleWidth } from "@earendil-works/pi-tui";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const PROVIDER_NAME = "Llama.cpp";
const API_KEY_PLACEHOLDER = "sk-placeholder";
const MODELS_JSON = join(process.env.HOME || ".", ".pi", "agent", "models.json");
const METADATA_JSON = join(process.env.HOME || ".", ".pi", "agent", "llama-metadata.json");

// ── Server Configs ────────────────────────────────────────────────────

interface ServerConfig {
  id: string;
  name: string;
  url: string;
}

interface ServerInfo {
  server: ServerConfig;
  ready: boolean;
  models: ModelsDataProperty[];
  mode?: ServerMode;
}

const PROVIDER_IDS = ["llama-cpp", "llama-cpp-remote", "llama-server"];
const apiKeyCache = new Map<string, string>();
const RPC_TIMEOUT = 2000; // 2s timeout for all server requests
const PROPS_TIMEOUT_MS = 120_000; // 2min timeout for /props (model loading can be slow)
const SETTING_KEY = "llamaStatusEnabled";

// ── Types ─────────────────────────────────────────────────────────────

interface ModelsDataProperty {
  id: string;
  aliases?: string[];
  status?: { value: string; args?: string[] };
  architecture?: { input_modalities: string[] };
  meta?: { n_ctx: number; n_ctx_train: number };
}

interface ModelsResponse {
  models?: ModelsDataProperty[]; // present in single mode, absent in router mode
  data: ModelsDataProperty[];
}

type ServerMode = "single" | "router";

function detectMode(res: ModelsResponse): ServerMode {
  return res.models ? "single" : "router";
}

interface PropsResponse {
  error?: { code: number; message: string };
  is_sleeping: boolean;
  default_generation_settings?: {
    n_ctx: number;
    params?: Record<string, any>;
  };
  total_slots?: number;
  model_path?: string;
  modalities?: { vision: boolean };
  build_info?: string;
}

interface SlotInfo {
  id: number;
  is_processing: boolean;
  n_ctx: number;
  next_token?: {
    has_next_token: boolean;
    n_decoded: number;
    n_remain: number;
  };
  params?: {
    temperature?: number;
    top_p?: number;
    n_predict?: number;
    samplers?: string[];
  };
}

interface MetricsData {
  kv_cache_usage_ratio: number | null;
  kv_cache_tokens: number | null;
  prompt_tokens_total: number | null;
  predicted_tokens_total: number | null;
  prompt_tokens_per_second: number | null;
  predicted_tokens_per_second: number | null;
  requests_processing: number | null;
  requests_deferred: number | null;
}

interface V1ModelMeta {
  vocab_type?: number;
  n_vocab?: number;
  n_ctx_train?: number;
  n_embd?: number;
  n_params?: number;
  size?: number;
}

interface V1ModelInfo {
  id: string;
  meta?: V1ModelMeta | null;
}

interface V1ModelsResponse {
  data: V1ModelInfo[];
}

interface ModelsJson {
  providers: Record<string, any>;
}

// ── Thinking Template Support ─────────────────────────────────────────

// Thinking budget (tokens) mapped from Pi thinking levels.
// Injected as thinking_budget_tokens in the request body for llama-cpp providers.
// off: rely on enable_thinking: false (no budget injection)
// xhigh: unrestricted (server default -1, no budget injection)
const THINKING_BUDGET_MAP: Record<string, number | undefined> = {
  off: undefined,
  low: 512,
  medium: 2048,
  high: 8192,
  xhigh: undefined,
};

// Qwen-style: chat_template_kwargs.enable_thinking (boolean toggle).
// Pi's qwen-chat-template format sends enable_thinking based on reasoning effort.
// String values expose levels in UI; Pi sends enable_thinking: false for off level.
// Granularity is controlled by thinking_budget_tokens injection in before_provider_request hook.
const QWEN_THINKING_LEVEL_MAP = {
  off: "off",
  minimal: null,
  low: "on",
  medium: "on",
  high: "on",
  xhigh: "on",
} satisfies NonNullable<ProviderModelConfig["thinkingLevelMap"]>;

// DeepSeek-style: chat_template_kwargs.thinking (effort string).
// Maps every Pi level to a distinct effort value via chatTemplateKwargs.
const CHAT_TEMPLATE_THINKING_LEVEL_MAP = {
  off: null,
  minimal: "minimal",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "max",
} satisfies NonNullable<ProviderModelConfig["thinkingLevelMap"]>;

// Apply Qwen-style thinking (enable_thinking boolean toggle).
function applyQwenThinkingSupport(model: Record<string, any>): void {
  model.reasoning = true;
  model.thinkingLevelMap = QWEN_THINKING_LEVEL_MAP;
  model.compat = {
    ...model.compat,
    // Despite the Pi enum name, this sends llama.cpp's generic
    // chat_template_kwargs.enable_thinking payload, not a Qwen-only option.
    thinkingFormat: "qwen-chat-template",
  };
}

// Apply chat-template style thinking (thinking effort string + chatTemplateKwargs).
function applyChatTemplateThinkingSupport(model: Record<string, any>): void {
  model.reasoning = true;
  model.thinkingLevelMap = CHAT_TEMPLATE_THINKING_LEVEL_MAP;
  model.compat = {
    ...model.compat,
    thinkingFormat: "chat-template",
    chatTemplateKwargs: {
      "thinking": { "$var": "thinking.effort", omitWhenOff: true },
    },
  };
}

// ── Config Resolution ─────────────────────────────────────────────────

let cachedSettings: Record<string, any> | undefined;
let modelsWriteTimer: NodeJS.Timeout | null = null;
let metadataWriteTimer: NodeJS.Timeout | null = null;

function loadSettings(): Record<string, any> {
  if (cachedSettings !== undefined) return cachedSettings;
  const path = join(process.env.HOME || ".", ".pi", "agent", "settings.json");
  if (existsSync(path)) {
    try { return JSON.parse(readFileSync(path, "utf-8")); } catch {}
  }
  return (cachedSettings = {});
}

function resolveLocalUrl(): string {
  const envOverride = process.env.LLAMA_SERVER_URL;
  const settings = loadSettings();
  const settingsOverride = settings?.llamaServerUrl;

  return (envOverride || settingsOverride || "http://127.0.0.1:8080").replace(/\/+$/, "");
}

function resolveRemoteUrl(): string | undefined {
  const settings = loadSettings();
  const raw = settings?.llamaServerRemoteUrl;
  if (!raw) return undefined;
  return raw.replace(/\/+$/, "");
}

function resolveServers(): ServerConfig[] {
  const localUrl = resolveLocalUrl();
  const servers: ServerConfig[] = [
    { id: "llama-cpp", name: `Local (${localUrl.replace("http://", "")})`, url: localUrl },
  ];

  const remoteUrl = resolveRemoteUrl();
  if (remoteUrl) {
    servers.push({
      id: "llama-cpp-remote",
      name: `Remote (${remoteUrl.replace("http://", "")})`,
      url: remoteUrl,
    });
  }

  return servers;
}

function resolveApiKeyFromDisk(serverId: string): string {
  const authPath = join(process.env.HOME || ".", ".pi", "agent", "auth.json");
  if (!existsSync(authPath)) return API_KEY_PLACEHOLDER;
  try {
    const cfg = JSON.parse(readFileSync(authPath, "utf-8"));
    // Try server-specific key first, then any known provider key
    if (cfg?.[serverId]?.key) return cfg[serverId].key;
    for (const id of PROVIDER_IDS) {
      if (cfg?.[id]?.key) return cfg[id].key;
    }
    return API_KEY_PLACEHOLDER;
  } catch {
    return API_KEY_PLACEHOLDER;
  }
}

function resolveApiKey(serverId: string): string {
  if (apiKeyCache.has(serverId)) {
    return apiKeyCache.get(serverId)!;
  }
  const key = resolveApiKeyFromDisk(serverId);
  apiKeyCache.set(serverId, key);
  return key;
}

function isLlamaStatusEnabled(): boolean {
  const settings = loadSettings();
  return settings[SETTING_KEY] !== false; // default true
}

// ── HTTP Client (per-server) ──────────────────────────────────────────

async function rpc<T>(server: ServerConfig, endpoint: string, body?: Record<string, unknown>): Promise<T> {
  const url = `${server.url}${endpoint}`;
  const apiKey = resolveApiKey(server.id);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), RPC_TIMEOUT);

  try {
    const res = await fetch(url, {
      method: body ? "POST" : "GET",
      headers: {
        ...(body ? { "Content-Type": "application/json" } : {}),
        ...(apiKey && apiKey !== API_KEY_PLACEHOLDER ? { Authorization: `Bearer ${apiKey}` } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    const text = await res.text();
    try { return JSON.parse(text); } catch {
      throw new Error(`Invalid JSON from ${endpoint}`);
    }
  } finally {
    clearTimeout(timer);
  }
}

// ── Additional Endpoints ──────────────────────────────────────────────

async function fetchSlots(server: ServerConfig, modelId?: string): Promise<SlotInfo[]> {
  const qs = modelId ? `?model=${encodeURIComponent(modelId)}` : "";
  try {
    return await rpc<SlotInfo[]>(server, `/slots${qs}`);
  } catch {
    return [];
  }
}

function parsePrometheusMetrics(text: string): MetricsData {
  const metrics: MetricsData = {
    kv_cache_usage_ratio: null,
    kv_cache_tokens: null,
    prompt_tokens_total: null,
    predicted_tokens_total: null,
    prompt_tokens_per_second: null,
    predicted_tokens_per_second: null,
    requests_processing: null,
    requests_deferred: null,
  };

  const map: Record<string, keyof MetricsData> = {
    "llamacpp:kv_cache_usage_ratio": "kv_cache_usage_ratio",
    "llamacpp:kv_cache_tokens": "kv_cache_tokens",
    "llamacpp:prompt_tokens_total": "prompt_tokens_total",
    "llamacpp:tokens_predicted_total": "predicted_tokens_total",
    "llamacpp:prompt_tokens_seconds": "prompt_tokens_per_second",
    "llamacpp:predicted_tokens_seconds": "predicted_tokens_per_second",
    "llamacpp:requests_processing": "requests_processing",
    "llamacpp:requests_deferred": "requests_deferred",
  };

  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.startsWith("#") || !trimmed) continue;
    for (const [name, key] of Object.entries(map)) {
      if (trimmed.startsWith(name)) {
        const lastSpace = trimmed.lastIndexOf(" ");
        const value = parseFloat(trimmed.slice(lastSpace + 1));
        if (!isNaN(value)) {
          (metrics as any)[key] = value;
        }
        break;
      }
    }
  }
  return metrics;
}

async function fetchMetrics(server: ServerConfig, modelId?: string): Promise<MetricsData> {
  const qs = modelId ? `?model=${encodeURIComponent(modelId)}` : "";
  try {
    const url = `${server.url}/metrics${qs}`;
    const apiKey = resolveApiKey(server.id);
    const res = await fetch(url, {
      headers: apiKey && apiKey !== API_KEY_PLACEHOLDER ? { Authorization: `Bearer ${apiKey}` } : {},
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    return parsePrometheusMetrics(text);
  } catch {
    return {
      kv_cache_usage_ratio: null, kv_cache_tokens: null,
      prompt_tokens_total: null, predicted_tokens_total: null,
      prompt_tokens_per_second: null, predicted_tokens_per_second: null,
      requests_processing: null, requests_deferred: null,
    };
  }
}

async function fetchV1Models(server: ServerConfig): Promise<V1ModelInfo[]> {
  try {
    const res = await rpc<V1ModelsResponse>(server, "/v1/models");
    return res.data || [];
  } catch {
    return [];
  }
}

async function loadModel(server: ServerConfig, modelId: string): Promise<void> {
  await rpc(server, "/models/load", { model: modelId });
}

// ── Model Inspector ───────────────────────────────────────────────────

class ModelInspector {
  private cachedData: ModelsDataProperty[] | null = null;
  private cachedMode: ServerMode | null = null;
  private cachedProps: PropsResponse | null = null;
  private cachedSlots: SlotInfo[] | null = null;
  private cachedMetrics: MetricsData | null = null;
  private cachedV1Models: V1ModelInfo[] | null = null;

  constructor(
    private server: ServerConfig,
    preloaded?: { data: ModelsDataProperty[]; mode: ServerMode },
  ) {
    if (preloaded) {
      this.cachedData = preloaded.data;
      this.cachedMode = preloaded.mode;
    }
  }

  private async fetchData(): Promise<ModelsDataProperty[]> {
    if (!this.cachedData) {
      const res = await rpc<ModelsResponse>(this.server, "/models");
      this.cachedData = res.data || [];
      this.cachedMode = detectMode(res);
    }
    return this.cachedData;
  }

  private getMode(): ServerMode {
    if (!this.cachedMode) throw new Error("Data not loaded");
    return this.cachedMode;
  }

  async list(): Promise<ModelsDataProperty[]> {
    if (!this.cachedData) await this.fetchData();
    return this.cachedData!;
  }

  async status(modelId: string): Promise<string> {
    const data = await this.fetchData();
    const model = data.find((m) => m.id === modelId);
    if (!model) return "failed";

    // Router mode: status from /models data
    if (this.getMode() === "router" && model.status?.value) {
      return model.status.value;
    }
    // Single mode: /props is server-wide, cache it
    const props = await this.getProps();
    if (props.is_sleeping) return "sleeping";
    if (!props.error) return "loaded";
    if (props.error.code === 503) return "loading";
    if (props.error.code === 400 && props.error.message === "model is not loaded") return "unloaded";
    return model?.status?.value || "failed";
  }

  private async getProps(): Promise<PropsResponse> {
    if (!this.cachedProps) {
      try {
        this.cachedProps = await rpc<PropsResponse>(this.server, "/props");
      } catch {
        this.cachedProps = { is_sleeping: false };
      }
    }
    return this.cachedProps;
  }

  contextSize(modelId: string): number {
    const model = this.cachedData?.find((m) => m.id === modelId);
    return model ? resolveContextSize(model) : 32768;
  }

  capabilities(modelId: string): string[] {
    const model = this.cachedData?.find((m) => m.id === modelId);
    if (!model?.architecture) return ["text"];
    return (model.architecture.input_modalities || ["text"]).filter(
      (m) => m === "text" || m === "image",
    );
  }

  async loadedModels(): Promise<Array<{ id: string; name: string; status: string }>> {
    const data = await this.fetchData();
    // Router mode: status is in /models data
    if (this.getMode() === "router") {
      const loaded: Array<{ id: string; name: string; status: string }> = [];
      for (const model of data) {
        const value = model.status?.value;
        if (value === "loaded" || value === "sleeping") {
          loaded.push({ id: model.id, name: model.aliases?.[0] || model.id, status: value });
        }
      }
      return loaded;
    }
    // Single mode: check /props once for the server
    const props = await this.getProps();
    if (props.is_sleeping) {
      const model = data[0];
      return [{ id: model.id, name: model.aliases?.[0] || model.id, status: "sleeping" }];
    }
    if (!props.error) {
      const model = data[0];
      return [{ id: model.id, name: model.aliases?.[0] || model.id, status: "loaded" }];
    }
    return [];
  }

  // ── Slots ───────────────────────────────────────────────────────────

  async getSlots(modelId?: string): Promise<SlotInfo[]> {
    if (!this.cachedSlots) {
      this.cachedSlots = await fetchSlots(this.server, modelId);
    }
    return this.cachedSlots;
  }

  getSlotInfo(modelId?: string): { decoded: number; remain: number; totalSlots: number; activeSlots: number } {
    const slots = this.cachedSlots || [];
    const active = slots.filter((s) => s.is_processing);
    let decoded = 0, remain = -1;
    for (const s of active) {
      if (s.next_token) {
        decoded += s.next_token.n_decoded;
        if (s.next_token.n_remain > 0) remain = s.next_token.n_remain;
      }
    }
    return { decoded, remain, totalSlots: slots.length, activeSlots: active.length };
  }

  // ── Metrics ─────────────────────────────────────────────────────────

  async getMetrics(modelId?: string): Promise<MetricsData> {
    if (!this.cachedMetrics) {
      this.cachedMetrics = await fetchMetrics(this.server, modelId);
    }
    return this.cachedMetrics;
  }

  // ── V1 Models (rich metadata) ──────────────────────────────────────

  async getV1Models(): Promise<V1ModelInfo[]> {
    if (!this.cachedV1Models) {
      this.cachedV1Models = await fetchV1Models(this.server);
    }
    return this.cachedV1Models;
  }

  async getModelMeta(modelId: string): Promise<V1ModelMeta | null> {
    const v1Models = await this.getV1Models();
    const m = v1Models.find((v) => v.id === modelId || v.id.endsWith(modelId));
    return m?.meta || null;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────

function isAutoExposedCacheEntry(m: ModelsDataProperty): boolean {
  // Auto-exposed HF cache entries have ID == hf-repo arg value
  const args = m.status?.args;
  if (args) {
    const idx = args.indexOf("--hf-repo");
    if (idx !== -1 && args[idx + 1] === m.id) return true;
  }
  return false;
}

function resolveContextSize(m: ModelsDataProperty): number {
  // Router mode: parse from status.args (--ctx-size or --fit-ctx)
  if (m.status?.args) {
    const args = m.status.args;
    for (const flag of ["--ctx-size", "--fit-ctx"]) {
      const idx = args.indexOf(flag);
      if (idx !== -1 && args[idx + 1]) {
        const parsed = parseInt(args[idx + 1], 10);
        if (!isNaN(parsed)) return parsed;
      }
    }
  }
  // Single mode: use meta.n_ctx
  if (m.meta?.n_ctx) return m.meta.n_ctx;
  // Fallback default
  return 32768;
}

function findServerByProvider(provider: string, servers: ServerConfig[]): ServerConfig | undefined {
  return servers.find((s) => s.id === provider);
}

// ── Server Gathering ──────────────────────────────────────────────────

async function gatherServers(): Promise<ServerInfo[]> {
  const servers = resolveServers();
  const serverInfo = await Promise.all(servers.map(async (server) => {
    let models: ModelsDataProperty[] = [];
    let mode: ServerMode | undefined;
    let ready = false;
    try {
      const res = await rpc<ModelsResponse>(server, "/models");
      models = res.data || [];
      mode = detectMode(res);
      ready = true;
    } catch {}
    return { server, ready, models, mode };
  }));

  return serverInfo;
}

// ── Status Indicator ──────────────────────────────────────────────────

const OVERLAY_WIDTH = 70;
const STATUS_ICONS: Record<string, string> = {
  loaded: "🟢",
  loading: "🟡",
  sleeping: "🔵",
  unloaded: "⚪",
  failed: "🔴",
  offline: "⬛",
};

function buildBorderDynamic(theme: Theme, lines: string[], boxWidth: number): string[] {
  const innerW = boxWidth - 2;
  const pad = (s: string) => s + " ".repeat(Math.max(0, innerW - visibleWidth(s)));
  const row = (content: string) =>
    theme.fg("border", "│") + pad(` ${content}`) + theme.fg("border", "│");
  const hr = () =>
    theme.fg("border", "│") + theme.fg("dim", "─".repeat(innerW)) + theme.fg("border", "│");

  const result: string[] = [];
  result.push(theme.fg("border", `╭${"─".repeat(innerW)}╮`));
  for (const line of lines) {
    if (line === "---") result.push(hr());
    else if (line === "") result.push(row(""));
    else result.push(row(line));
  }
  result.push(theme.fg("border", `╰${"─".repeat(innerW)}╯`));
  return result;
}

function formatParams(n: number | undefined): string {
  if (!n) return "?";
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return n.toString();
}

function formatBytes(bytes: number | undefined): string {
  if (!bytes) return "?";
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(1)} GB`;
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(0)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

function formatMetrics(m: MetricsData): string[] {
  const parts: string[] = [];
  if (m.kv_cache_usage_ratio !== null) {
    parts.push(`KV Cache: ${(m.kv_cache_usage_ratio * 100).toFixed(1)}%`);
  }
  if (m.kv_cache_tokens !== null) {
    parts.push(`${m.kv_cache_tokens.toLocaleString()} cached`);
  }
  if (m.predicted_tokens_per_second !== null) {
    parts.push(`Gen: ${m.predicted_tokens_per_second.toFixed(1)} tok/s`);
  }
  if (m.prompt_tokens_per_second !== null) {
    parts.push(`Prefill: ${m.prompt_tokens_per_second.toFixed(1)} tok/s`);
  }
  if (m.requests_processing !== null && m.requests_processing > 0) {
    parts.push(`${m.requests_processing} processing`);
  }
  if (m.requests_deferred !== null && m.requests_deferred > 0) {
    parts.push(`${m.requests_deferred} deferred`);
  }
  return parts;
}

async function buildStatusLines(current: ProviderModelConfig | undefined): Promise<string[]> {
  const serverInfo = await gatherServers();
  const currentProvider = (current as any)?.provider;
  const isLlamaModel = current && PROVIDER_IDS.includes(currentProvider);
  const lines: string[] = [];

  for (const { server, ready, models, mode } of serverInfo) {
    // Separate from previous server block
    if (lines.length > 0) lines.push("");
    lines.push(`${server.name}${ready ? "" : " — ⬛ offline"}`);

    if (!ready) continue;

    const inspector = new ModelInspector(server, models.length > 0 ? { data: models, mode: mode! } : undefined);
    const loadedModels = await inspector.loadedModels();

    // Sort so the active Pi model comes first
    loadedModels.sort((a, b) => {
      const aActive = isLlamaModel && currentProvider === server.id && current.id === a.id ? 1 : 0;
      const bActive = isLlamaModel && currentProvider === server.id && current.id === b.id ? 1 : 0;
      return bActive - aActive;
    });

    if (loadedModels.length === 0) {
      lines.push(`  ⚪ No model loaded`);
    }

    for (const serverModel of loadedModels) {
      const { id, name, status } = serverModel;
      const icon = STATUS_ICONS[status] || "⚪";
      const contextSize = inspector.contextSize(id);
      const caps = inspector.capabilities(id);
      const isActive = isLlamaModel && currentProvider === server.id && current.id === id;
      const isSleeping = status === "sleeping";

      lines.push(`  ${icon} ${name} (${status})${isActive ? " ✓ active" : ""}`);
      lines.push(`     Context: ${contextSize.toLocaleString()} tokens · Input: ${caps.join(", ")}`);

      // Skip live endpoints for sleeping models — they wake the model on the router
      if (!isSleeping) {
        // Fetch slots, metrics, v1 metadata in parallel
        const [meta, slots, metrics] = await Promise.all([
          inspector.getModelMeta(id),
          inspector.getSlots(mode === "router" ? id : undefined),
          inspector.getMetrics(mode === "router" ? id : undefined),
        ]);

        // Model metadata
        if (meta) {
          const parts: string[] = [];
          if (meta.n_params) parts.push(`${formatParams(meta.n_params)} params`);
          if (meta.n_vocab) parts.push(`${formatParams(meta.n_vocab)} vocab`);
          if (meta.size) parts.push(`${formatBytes(meta.size)}`);
          if (meta.n_ctx_train) parts.push(`Train ctx: ${meta.n_ctx_train.toLocaleString()}`);
          if (parts.length) {
            lines.push(`     ${parts.join(" · ")}`);
          }
        }

        // Slot info (always shown when slots endpoint is available)
        const slotInfo = inspector.getSlotInfo(mode === "router" ? id : undefined);
        if (slotInfo.totalSlots > 0) {
          const genInfo: string[] = [`${slotInfo.activeSlots}/${slotInfo.totalSlots} slots`];
          if (slotInfo.decoded > 0) genInfo.push(`${slotInfo.decoded} tokens`);
          if (slotInfo.remain > 0) genInfo.push(`${slotInfo.remain} remaining`);
          lines.push(`     ▶ ${genInfo.join(" · ")}`);
        }

        // Metrics
        const metricLines = formatMetrics(metrics);
        if (metricLines.length) {
          lines.push(`     📊 ${metricLines.join(" · ")}`);
        }
      }
    }

    const allModels = await inspector.list();
    const filteredModels = allModels.filter((m) => !isAutoExposedCacheEntry(m));
    if (filteredModels.length > 0) {
      lines.push(`  Models:`);
      for (const m of filteredModels) {
        const status = await inspector.status(m.id);
        const icon = STATUS_ICONS[status] || "⚪";
        const name = m.aliases?.[0] || m.id;
        const active = isLlamaModel && currentProvider === server.id && current.id === m.id ? " ← active" : "";
        lines.push(`    ${icon} ${name}${active}`);
      }
    }
  }

  return lines;
}

async function showStatus(ctx: ExtensionCommandContext): Promise<void> {
  const contentLines = await buildStatusLines(ctx.model);

  await ctx.ui.custom<void>(
    (tui, theme, _keybindings, done) => {
      let currentWidth = OVERLAY_WIDTH;
      return {
        handleInput(data: string) {
          if (matchesKey(data, "escape") || matchesKey(data, "q")) {
            done(undefined);
          }
        },
        render(width: number): string[] {
          const w = Math.max(OVERLAY_WIDTH, width);
          if (w !== currentWidth) currentWidth = w;
          const overlayLines = [
            theme.bold(theme.fg("accent", `${PROVIDER_NAME} Status`)),
            "",
            ...contentLines,
            "",
            "---",
            "",
            "Press Escape or q to close",
          ];
          return buildBorderDynamic(theme, overlayLines, currentWidth);
        },
        invalidate() {},
      };
    },
    {
      overlay: true,
      overlayOptions: {
        anchor: "center",
        width: "80%",
        // maxWidth: OVERLAY_WIDTH,
        minWidth: 50,
        maxHeight: "90%",
      },
    },
  );
}

// ── Unload Command ────────────────────────────────────────────────────

async function unloadModel(ctx: ExtensionCommandContext): Promise<void> {
  const current = ctx.model;

  const modelProvider = (current as any)?.provider;
  if (!current || !PROVIDER_IDS.includes(modelProvider)) {
    ctx.ui.notify(`Current model is not ${PROVIDER_NAME} (provider: ${modelProvider || "none"})`, "error");
    return;
  }

  const servers = resolveServers();
  const server = findServerByProvider(modelProvider, servers);

  if (!server) {
    ctx.ui.notify(`No server found for provider ${modelProvider}`, "error");
    return;
  }

  let modelsRes: ModelsResponse;
  try {
    modelsRes = await rpc<ModelsResponse>(server, "/models");
  } catch {
    ctx.ui.notify(`${server.name} unreachable`, "error");
    return;
  }
  const mode = detectMode(modelsRes);
  const inspector = new ModelInspector(server, { data: modelsRes.data || [], mode });
  const loadedModels = await inspector.loadedModels();
  // Prefer the current Pi model; fall back to first loaded
  const serverModel = loadedModels.find((m) => m.id === current.id) || loadedModels[0];

  if (!serverModel) {
    ctx.ui.notify(`${server.name}: no model loaded`, "info");
    return;
  }

  const modelId = mode === "router" ? serverModel.id : current.id;
  try {
    await rpc(server, "/models/unload", { model: modelId });
    ctx.ui.notify(`Unloaded ${serverModel.name} from ${server.name}`, "info");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    ctx.ui.notify(`Failed to unload: ${msg}`, "error");
  }
}

async function loadModelCmd(ctx: ExtensionCommandContext, modelArg: string): Promise<void> {
  const servers = resolveServers();
  const server = servers[0]; // Use local server by default

  let ready = false;
  try {
    await rpc<ModelsResponse>(server, "/models");
    ready = true;
  } catch {
    ctx.ui.notify(`${server.name} unreachable`, "error");
    return;
  }

  // Start SSE listener to pick up loading progress
  if (!isSseActive()) {
    startSseForServer(server.id, ctx);
  }

  // If a model ID was provided directly, load it
  if (modelArg) {
    try {
      await loadModel(server, modelArg);
      ctx.ui.notify(`Loading ${modelArg} on ${server.name}`, "info");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("already running") || msg.includes("already loaded")) {
        ctx.ui.notify(`${modelArg} is already loaded`, "info");
      } else {
        ctx.ui.notify(`Failed to load: ${msg}`, "error");
      }
    }
    return;
  }

  // No model specified — show selection UI
  const inspector = new ModelInspector(server);
  const models = (await inspector.list()).filter((m) => !isAutoExposedCacheEntry(m));

  if (models.length === 0) {
    ctx.ui.notify("No models available on server", "error");
    return;
  }

  const options = models.map((m) => {
    const name = m.aliases?.[0] || m.id;
    const status = m.status?.value || "unknown";
    const icon = STATUS_ICONS[status] || "⚪";
    return `${icon} ${name}`;
  });

  const choice = await ctx.ui.select(`Load model on ${server.name}:`, options);
  if (!choice) return;

  const selectedIndex = options.indexOf(choice);
  const selected = models[selectedIndex];
  if (!selected) {
    ctx.ui.notify("Model not found", "error");
    return;
  }

  const displayName = selected.aliases?.[0] || selected.id;
  try {
    await loadModel(server, selected.id);
    ctx.ui.notify(`Loading ${displayName} on ${server.name}`, "info");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes("already running") || msg.includes("already loaded")) {
      ctx.ui.notify(`${displayName} is already loaded`, "info");
    } else {
      ctx.ui.notify(`Failed to load: ${msg}`, "error");
    }
  }
}

// ── Sync to models.json ──────────────────────────────────────────────

function loadModelsJson(): ModelsJson {
  if (existsSync(MODELS_JSON)) {
    try { return JSON.parse(readFileSync(MODELS_JSON, "utf-8")); } catch {}
  }
  return { providers: {} };
}

function modelsChanged(
  existing: any[],
  incoming: Array<{ id: string; contextWindow: number; reasoning?: boolean }>,
): boolean {
  if (existing.length !== incoming.length) return true;
  const existingMap = new Map(existing.map((m: any) => [m.id, m]));
  for (const m of incoming) {
    const match = existingMap.get(m.id);
    if (!match) return true;
    if (match.contextWindow !== m.contextWindow) return true;
    if (m.reasoning !== match.reasoning) return true;
  }
  return false;
}

async function syncToModelsJson(): Promise<boolean> {
  const serverInfo = await gatherServers();
  const config = loadModelsJson();
  let wrote = false;
  const validModels = new Map<string, Set<string>>();

  for (const { server, ready, models } of serverInfo) {
    if (!ready) continue;

    // Filter out auto-exposed HF cache entries (undefined models)
    const filteredModels = models.filter((m) => !isAutoExposedCacheEntry(m));
    validModels.set(server.id, new Set(filteredModels.map((m) => m.id)));
    if (filteredModels.length === 0) continue;

    const modelConfigs: ProviderModelConfig[] = filteredModels.map(m => {
      const contextWindow = resolveContextSize(m);
      return {
        id: m.id,
        name: m.aliases?.[0] || m.id,
        input: (m.architecture?.input_modalities || ["text"]).filter(
          (mod) => mod === "text" || mod === "image",
        ),
        contextWindow,
        maxTokens: Math.min(32000, contextWindow),
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      };
    });

    const modelsWithOverlay = modelConfigs.map(m => {
      const { id, name, input, contextWindow, maxTokens, cost } = m;
      const result: any = { id, name, input, contextWindow, maxTokens, cost };
      applyMetadataOverlay(result, server.id);
      return result;
    });

    const existing = config.providers[server.id]?.models || [];
    if (!modelsChanged(existing, modelsWithOverlay)) continue;

    config.providers[server.id] = {
      baseUrl: server.url + "/v1",
      api: "openai-completions",
      apiKey: resolveApiKey(server.id),
      models: modelsWithOverlay.map(m => ({ ...m, reasoning: m.reasoning ?? false })),
    };
    wrote = true;
  }

  if (wrote) {
    if (modelsWriteTimer) clearTimeout(modelsWriteTimer);
    modelsWriteTimer = setTimeout(() => {
      writeFileSync(MODELS_JSON, JSON.stringify(config, null, 2) + "\n");
      modelsWriteTimer = null;
    }, 1000);
  }

  // Prune metadata for removed/renamed models (only for reachable servers)
  cleanupStaleMetadata(validModels, serverInfo.filter((s) => s.ready).map((s) => s.server.id));

  return wrote;
}

// ── Lazy /props Metadata Discovery ────────────────────────────────────

// Track which server:model combos have been discovered to avoid duplicate queries
const discoveredMetadata = new Set<string>();
const pendingMetadata = new Set<string>();

// ── SSE Model Loading Progress ────────────────────────────────────────

interface SseProgress {
  stages?: string[];   // ["text_model", "spec_model", "mmproj_model"]
  current?: string;    // "text_model", "spec_model", "mmproj_model"
  stage?: string;      // older format: single stage name
  value?: number;      // 0.0 — 1.0
}

interface ModelLoadState {
  status: string;      // "loading", "loaded", "sleeping", "unloaded"
  progress?: SseProgress;
  modelId?: string;
}

// Per-model load state tracked from SSE events
const loadStateMap = new Map<string, ModelLoadState>();

// Active SSE connection state
let sseAbort: AbortController | null = null;
let sseServerId: string | "" = "";
let sseCtx: ExtensionContext | null = null;
let sseReconnectTimer: NodeJS.Timeout | null = null;
let sseReconnectAttempts = 0;
const SSE_MAX_RECONNECT_ATTEMPTS = 10;
const SSE_INITIAL_RECONNECT_MS = 1000;

/** Stage name → human-readable label */
const STAGE_LABELS: Record<string, string> = {
  "fit_params": "fitting params",
  "text_model": "model",
  "mmproj_model": "mmproj",
};

/** Format a stage name for display */
function formatStage(stage: string): string {
  return STAGE_LABELS[stage] || stage;
}

/**
 * Format the loading progress string for the status bar.
 * Matches tps/gallop style: dim prefix, accent for key values, dim for detail.
 */
function formatLoadingProgress(state: ModelLoadState, theme: any): string {
  const dim = (s: string) => theme.fg("dim", s);
  const accent = (s: string) => theme.fg("accent", s);
  const success = (s: string) => theme.fg("success", s);

  if (state.status === "loading" && state.progress) {
    const prog = state.progress;
    const stage = prog.current || prog.stage;
    const value = prog.value;

    if (stage && value !== undefined) {
      const pct = Math.round(value * 100);
      if (stage === "fit_params") {
        return `${dim("· ")}${accent("Loading")} ${dim(`${formatStage(stage)}...`)}`;
      }
      return `${dim("· ")}${accent("Loading")} ${dim(`${formatStage(stage)} ${pct}%`)}`;
    }
  }

  if (state.status === "loading") {
    return `${dim("· ")}${accent("Loading")} ${dim("...")}`;
  }

  if (state.status === "loaded") {
    return `${success("✓")} ${dim("loaded")}`;
  }

  return "";
}

/**
 * Parse SSE stream from a Response body.
 * Yields parsed JSON objects from "data:" lines.
 */
async function* parseSseStream(response: Response): AsyncGenerator<string> {
  if (!response.body) return;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";

      for (const part of parts) {
        const lines = part.split("\n");
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data:")) {
            const json = trimmed.slice(5).trim();
            if (json) yield json;
          }
        }
      }
    }

    // Process remaining buffer
    if (buffer.trim()) {
      const lines = buffer.split("\n");
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data:")) {
          const json = trimmed.slice(5).trim();
          if (json) yield json;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Connect to /models/sse and process status_change events.
 * Reconnects with exponential backoff on disconnect (up to max attempts).
 */
async function connectSse(
  server: ServerConfig,
  ctx: ExtensionContext,
): Promise<void> {
  const apiKey = resolveApiKey(server.id);
  const url = `${server.url}/models/sse`;

  sseAbort = new AbortController();
  sseCtx = ctx;
  sseServerId = server.id;

  try {
    const response = await fetch(url, {
      headers: {
        "Accept": "text/event-stream",
        ...(apiKey && apiKey !== API_KEY_PLACEHOLDER
          ? { Authorization: `Bearer ${apiKey}` }
          : {}),
      },
      signal: sseAbort.signal,
    });

    if (!response.ok) {
      // SSE endpoint not available (e.g., single-model mode or old server)
      return;
    }

    sseReconnectAttempts = 0; // Reset reconnect counter on successful connect

    for await (const jsonStr of parseSseStream(response)) {
      try {
        const event = JSON.parse(jsonStr);
        handleSseEvent(event, server.id, ctx);
      } catch {
        // Skip malformed SSE data lines
      }
    }
  } catch (err: any) {
    const msg = err?.name || err?.message || String(err);
    // AbortError is expected when we intentionally disconnect
    if (msg === "AbortError" || msg === "aborted") return;

    // Server error or network failure — attempt reconnect
    attemptSseReconnect(server, ctx);
  }
}

/**
 * Attempt to reconnect SSE with exponential backoff.
 */
function attemptSseReconnect(
  server: ServerConfig,
  ctx: ExtensionContext,
): void {
  if (sseReconnectAttempts >= SSE_MAX_RECONNECT_ATTEMPTS) {
    return;
  }

  sseReconnectAttempts++;
  const delay = Math.min(
    SSE_INITIAL_RECONNECT_MS * Math.pow(2, sseReconnectAttempts - 1),
    30000, // Cap at 30s
  );

  sseReconnectTimer = setTimeout(() => {
    connectSse(server, ctx);
  }, delay);
}

/**
 * Handle a parsed SSE data line.
 * SSE format: data: {"model":"...","event":"status_change","data":{"status":"loading","progress":{...}}}
 * The JSON has: model, event, data (with status + progress inside).
 */
function handleSseEvent(
  payload: any,
  serverId: string,
  ctx: ExtensionContext,
): void {
  if (!payload || !payload.model) return;

  // payload.data contains {status, progress, ...}
  const inner = payload.data;
  if (!inner || !inner.status) return;

  const modelId = payload.model;
  const status = inner.status;
  const progress = inner.progress;

  // Track state per model
  const state: ModelLoadState = {
    status,
    progress: progress || undefined,
    modelId,
  };
  loadStateMap.set(modelId, state);

  // Update status bar if this model belongs to the active server
  if (serverId === sseServerId && sseCtx) {
    try {
      const theme = ctx.ui.theme;
      const progressStr = formatLoadingProgress(state, theme);
      if (progressStr) {
        ctx.ui.setStatus("llama", progressStr);
      }

      // Clear status bar when model is fully loaded
      if (status === "loaded") {
        const progressStr = formatLoadingProgress(state, theme);
        ctx.ui.setStatus("llama", progressStr);
        // Clear after a delay so user sees the loaded confirmation
        setTimeout(() => {
          if (sseCtx) {
            try { sseCtx.ui.setStatus("llama", undefined); } catch {}
          }
        }, 5000);
      }
    } catch {
      // Context may be stale after session end
    }
  }
}

/**
 * Stop the active SSE connection and clear state.
 */
function stopSse(): void {
  if (sseReconnectTimer) {
    clearTimeout(sseReconnectTimer);
    sseReconnectTimer = null;
  }
  if (sseAbort) {
    sseAbort.abort();
    sseAbort = null;
  }
  sseServerId = "";
  sseReconnectAttempts = 0;
}

/**
 * Start SSE listener for a server if not already connected.
 */
function startSseForServer(serverId: string, ctx: ExtensionContext): void {
  // If already connected to this server, keep it
  if (sseServerId === serverId) return;

  // Stop existing connection
  stopSse();

  const servers = resolveServers();
  const server = servers.find((s) => s.id === serverId);
  if (!server) return;

  // Start SSE connection
  connectSse(server, ctx);
}

/**
 * Check if SSE is connected to a llama-cpp provider.
 */
function isSseActive(): boolean {
  return sseServerId !== "" && PROVIDER_IDS.includes(sseServerId);
}

// ── Metadata Overlay ──────────────────────────────────────────────────
// Persists model capabilities (thinking, context size) per server:model so it survives model syncs.

interface ModelMetadata {
  [serverId: string]: {
    [modelId: string]: { thinking?: boolean | string; contextWindow?: number };
  };
}

function loadMetadataOverlay(): ModelMetadata {
  if (existsSync(METADATA_JSON)) {
    try { return JSON.parse(readFileSync(METADATA_JSON, "utf-8")); } catch {}
  }
  return {};
}

function saveMetadataOverlay(metadata: ModelMetadata): void {
  if (metadataWriteTimer) clearTimeout(metadataWriteTimer);
  metadataWriteTimer = setTimeout(() => {
    writeFileSync(METADATA_JSON, JSON.stringify(metadata, null, 2) + "\n");
    metadataWriteTimer = null;
  }, 1000);
}

function cleanupStaleMetadata(validModels: Map<string, Set<string>>, reachableServers: string[]): void {
  const overlay = loadMetadataOverlay();
  let pruned = false;
  for (const serverId of Object.keys(overlay)) {
    // Skip servers that weren't reachable — don't delete their metadata
    if (!reachableServers.includes(serverId)) continue;

    const valid = validModels.get(serverId);
    if (!valid) {
      delete overlay[serverId];
      pruned = true;
      continue;
    }
    for (const modelId of Object.keys(overlay[serverId])) {
      if (!valid.has(modelId)) {
        delete overlay[serverId][modelId];
        pruned = true;
      }
    }
    if (Object.keys(overlay[serverId]).length === 0) {
      delete overlay[serverId];
      pruned = true;
    }
  }
  if (pruned) {
    saveMetadataOverlay(overlay);
  }
}

function persistModelMetadata(serverId: string, modelId: string, data: { thinking?: boolean | string; contextWindow?: number }): void {
  const overlay = loadMetadataOverlay();
  if (!overlay[serverId]) overlay[serverId] = {};
  const existing = overlay[serverId][modelId] || {};
  overlay[serverId][modelId] = { ...existing, ...data };
  saveMetadataOverlay(overlay);
}

function applyMetadataOverlay(model: ProviderModelConfig, serverId: string): void {
  const overlay = loadMetadataOverlay();
  const entry = overlay[serverId]?.[model.id];
  if (!entry) return;
  if (entry.thinking) {
    if (entry.thinking === "chat-template") {
      applyChatTemplateThinkingSupport(model as Record<string, any>);
    } else {
      // Legacy boolean or "qwen-chat-template" → Qwen-style
      applyQwenThinkingSupport(model as Record<string, any>);
    }
  }
  if (entry.contextWindow) {
    model.contextWindow = entry.contextWindow;
    model.maxTokens = Math.min(32000, entry.contextWindow);
  }
}

async function discoverModelMetadata(
  pi: ExtensionAPI,
  serverId: string,
  modelId: string,
  ctx?: ExtensionContext,
  timeoutMs = PROPS_TIMEOUT_MS,
): Promise<void> {
  const servers = resolveServers();
  const server = servers.find((s) => s.id === serverId);
  if (!server) return;

  const key = `${serverId}:${modelId}`;
  if (discoveredMetadata.has(key)) return;
  if (pendingMetadata.has(key)) return;

  pendingMetadata.add(key);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const propsUrl = `${server.url.replace(/\/+$/, "")}/props?model=${encodeURIComponent(modelId)}&autoload=false`;

  // Safe ctx wrapper — session can be replaced after model switch, making ctx stale
  const u = (fn: (c: ExtensionContext) => void) => { try { if (ctx) fn(ctx); } catch {} };

  try {
    const response = await fetch(propsUrl, {
      signal: controller.signal,
      headers: {
        ...(resolveApiKey(serverId) !== API_KEY_PLACEHOLDER ? { Authorization: `Bearer ${resolveApiKey(serverId)}` } : {}),
      },
    });

    if (!response.ok) {
      u((c) => c.ui.notify(`[llama-cpp] /props for ${modelId} returned ${response.status}`, "error"));
      discoveredMetadata.add(key);
      return;
    }

    const data = await response.json();
    let updated = false;
    const metadata: { thinking?: boolean | string; contextWindow?: number } = {};

    if (data?.chat_template?.includes("enable_thinking") === true) {
      metadata.thinking = true; // Qwen-style boolean toggle
      updated = true;
    } else if (/\{[%{]\s*thinking\b/.test(data?.chat_template || "")) {
      // DeepSeek-style: chat_template_kwargs.thinking (effort string)
      // Matches {{ thinking }}, {% if thinking %}, {% set thinking = ..., etc.
      metadata.thinking = "chat-template";
      updated = true;
    }

    if (data?.default_generation_settings?.n_ctx) {
      metadata.contextWindow = data.default_generation_settings.n_ctx;
      updated = true;
    }

    if (Object.keys(metadata).length > 0) {
      persistModelMetadata(serverId, modelId, metadata);
    }

    discoveredMetadata.add(key);

    if (!updated) return;

    // Lazy re-sync to apply overlay to models.json without blocking
    void syncToModelsJson().catch(() => {});
  } catch (error) {
    const err = error as Error;
    const msg = err.name === "AbortError" ? "timeout" : err.message;
    u((c) => c.ui.setStatus(serverId, undefined));
    u((c) => c.ui.notify(`[llama-cpp] /props for ${modelId} failed: ${msg}`, "error"));
  } finally {
    clearTimeout(timer);
    pendingMetadata.delete(key);
  }
}

// ── Extension Entry ───────────────────────────────────────────────────

export default function llamaStatusExtension(pi: ExtensionAPI) {
  pi.registerCommand("llama-model", {
    description: `${PROVIDER_NAME} status indicator`,
    handler: async (_args: string, ctx: ExtensionCommandContext) => {
      await showStatus(ctx);
    },
  });

  pi.registerCommand("llama-unload", {
    description: `Unload current ${PROVIDER_NAME} model`,
    handler: async (_args: string, ctx: ExtensionCommandContext) => {
      await unloadModel(ctx);
    },
  });

  pi.registerCommand("llama-load", {
    description: `Load a ${PROVIDER_NAME} model (router mode)`,
    handler: async (args: string, ctx: ExtensionCommandContext) => {
      await loadModelCmd(ctx, args.trim());
    },
  });

  pi.registerCommand("llama-sync", {
    description: `Sync ${PROVIDER_NAME} models to models.json`,
    handler: async (_args: string, ctx: ExtensionCommandContext) => {
      const wrote = await syncToModelsJson();
      ctx.ui.notify(wrote ? `${PROVIDER_NAME} models synced` : `${PROVIDER_NAME} models already up to date`, "info");
    },
  });

  // Auto-sync on session start
  pi.on("session_start", async (_event: any, ctx: ExtensionContext) => {
    if (!isLlamaStatusEnabled()) return;
    try { await syncToModelsJson(); } catch {}
  });

  // ── SSE Model Loading Progress ──────────────────────────────────────

  // Connect SSE listener early on session start to catch auto-load events
  pi.on("session_start", async (_event: any, ctx: ExtensionContext) => {
    if (!isLlamaStatusEnabled()) return;
    const provider = (ctx.model as any)?.provider;
    if (provider && PROVIDER_IDS.includes(provider) && !isSseActive()) {
      startSseForServer(provider, ctx);
    }
  });

  // Reconnect SSE if model changes during session
  pi.on("model_select", async (event: any, ctx: ExtensionContext) => {
    if (!isLlamaStatusEnabled()) return;
    const provider = (event.model as any)?.provider;
    if (!provider || !PROVIDER_IDS.includes(provider)) {
      // Switched away from llama-cpp — stop SSE
      stopSse();
      return;
    }

    // If SSE is already active for this provider, keep it
    if (isSseActive() && sseServerId === provider) return;

    startSseForServer(provider, ctx);
  });

  // Clean up SSE connection on session shutdown
  pi.on("session_shutdown", async () => {
    stopSse();
    sseCtx = null;
    loadStateMap.clear();
  });

  // ── Additional Commands ─────────────────────────────────────────────

  pi.registerCommand("llama-version", {
    description: "Print llama-server --version output",
    handler: async (_args: string, ctx: ExtensionCommandContext) => {
      const result = await pi.exec("llama-server", ["--version"]);
      const output = `${result.stderr ?? ""}\n${result.stdout ?? ""}`;
      const versionLine = output
        .split("\n")
        .map((l) => l.trim())
        .find((l) => /^version:\s/i.test(l));
      ctx.ui.notify(
        versionLine ?? `llama-server exited with code ${result.code}`,
        versionLine ? "info" : "error",
      );
    },
  });

  // ── Event Handlers ──────────────────────────────────────────────────

  // Inject thinking_budget_tokens for llama-cpp reasoning models
  pi.on("before_provider_request", (event, ctx) => {
    const provider = (ctx.model as any)?.provider;
    if (!PROVIDER_IDS.includes(provider || "")) return;
    if (!(ctx.model as any)?.reasoning) return;

    const level = pi.getThinkingLevel();
    const budget = THINKING_BUDGET_MAP[level];
    if (budget === undefined) return;

    return {
      ...event.payload,
      thinking_budget_tokens: budget,
    };
  });

  // /props metadata after first successful provider response
  // (model is guaranteed loaded by this point — no race with model loading)
  pi.on("after_provider_response", (event, ctx) => {
    if (event.status !== 200) return;
    const provider = (ctx.model as any)?.provider;
    if (!PROVIDER_IDS.includes(provider || "")) return;
    void discoverModelMetadata(
      pi,
      provider,
      ctx.model!.id,
      ctx,
      PROPS_TIMEOUT_MS,
    );
  });

  pi.registerCommand("llama-status", {
    description: "Toggle llama-status extension on/off",
    handler: async (_args, ctx) => {
      const settings = loadSettings();
      settings[SETTING_KEY] = !settings[SETTING_KEY];
      mkdirSync(join(process.env.HOME || ".", ".pi", "agent"), { recursive: true });
      writeFileSync(join(process.env.HOME || ".", ".pi", "agent", "settings.json"), JSON.stringify(settings, null, 2) + "\n");
      ctx.ui.notify(settings[SETTING_KEY] ? "Llama status enabled" : "Llama status disabled", settings[SETTING_KEY] ? "info" : "warning");
    },
  });

}
