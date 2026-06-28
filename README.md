# llama-status

Llama.cpp server status indicator, model load/unload, and `models.json` sync for Pi.

## Commands

| Command | Description |
|---------|-------------|
| `/llama-model` | Overlay popup showing server status, model metadata, slots, metrics, and available models |
| `/llama-unload` | Unload the current model if it's from a llama.cpp provider |
| `/llama-load` | Open model picker to load a model (router mode) |
| `/llama-load <id>` | Load a specific model by ID (router mode) |
| `/llama-sync` | Manually sync all server models to `models.json` |
| `/llama-version` | Print `llama-server --version` output |
| `/llama-status` | Toggle llama-status extension on/off |

## Servers

One local server always present. Remote server is opt-in.

| Provider ID | Default | Configurable |
|-------------|---------|--------------|
| `llama-cpp` | `http://127.0.0.1:8080` | Yes (see below) |
| `llama-cpp-remote` | None (opt-in) | Yes (`llamaServerRemoteUrl`) |

### URL Resolution (local server)

Priority order: `.pi/llama-server.json` → `LLAMA_SERVER_URL` env → `settings.json` (`llamaServerUrl`) → `127.0.0.1:8080`.

### URL Resolution (remote server)

`settings.json` (`llamaServerRemoteUrl`) → omitted if not set. Set to `""` to explicitly disable.

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `llamaStatusEnabled` | `true` | Toggle extension on/off via `/llama-status` |
| `llamaServerUrl` | `http://127.0.0.1:8080` | Local server URL (overridden by env var) |
| `llamaServerRemoteUrl` | None | Remote server URL (opt-in) |

## SSE Loading Progress

Connects to `/models/sse` to show real-time loading progress in the status bar:

- **Stages**: `fit_params` → `text_model` → `mmproj_model` (for vision models)
- **Display**: `Loading model 42%`, `Loading mmproj 78%`, etc.
- **Reconnect**: Auto-reconnects with exponential backoff (up to 10 attempts, capped at 30s)
- **Cleanup**: SSE connection stops when session ends or model switches away from llama.cpp

## Auto-Sync

On `session_start`, syncs model metadata to `~/.pi/agent/models.json`.

- `id`, `name`, `input` (capabilities), `contextWindow`, `maxTokens`
- Skips write if model list and context windows are unchanged
- Each server writes under its own provider key
- Filters out auto-exposed HF cache entries (undefined models like `unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_XL`)
- Removes provider entries for servers no longer configured (e.g., remote URL unset)

## Thinking Support

Autodetects thinking capability from each model's chat template via `/props`:

- **Qwen-style** (`enable_thinking` variable) → boolean toggle, `thinkingFormat: "qwen-chat-template"`. Only off/medium exposed.
- **DeepSeek-style** (`thinking` variable) → effort string via `chatTemplateKwargs`, `thinkingFormat: "chat-template"`. Full level mapping (off/minimal/low/medium/high/xhigh).

Discovered metadata is persisted to `llama-metadata.json` and applied on every model sync.

## Architecture

- `rpc(server, endpoint, body?)` — per-server HTTP client (all API calls go through this)
- `isServerReady(server)` — `/health` check
- `fetchSlots(server, modelId?)` — `GET /slots` (or `/slots?model=X` in router mode)
- `fetchMetrics(server, modelId?)` — `GET /metrics` (Prometheus text format, parsed to MetricsData)
- `fetchV1Models(server)` — `GET /v1/models` for rich metadata (n_params, n_vocab, size)
- `loadModel(server, modelId)` — `POST /models/load` (router mode)
- `detectMode(res)` — `models` field present → single, absent → router
- `getModelStatus(server, id)` — router: from `/models` data; single: from `/props`. Returns `loaded|loading|sleeping|unloaded|failed`
- `resolveContextSize(model)` — router: parses `--ctx-size` then `--fit-ctx` from `status.args`; single: `meta.n_ctx`. Fallback: 32768.
- `buildStatusLines(cwd, current)` — gathers all data, returns plain string lines
- `buildBorderDynamic(theme, lines, width)` — wraps lines in box-drawing border using `visibleWidth()` for emoji-safe padding

## Status Display

The `/llama-model` overlay shows per-server:
- **Model info**: name, status, context size, input modalities
- **Metadata** (from `/v1/models`): params, vocab size, file size, training context
- **Active generation** (from `/slots`): active/total slots, tokens decoded, remaining
- **Metrics** (from `/metrics`, requires `--metrics` flag): KV cache %, gen/prefill tok/s, queue depth
- **Available models**: all registered models with status icons

## Gotchas

- **Emoji width**: use `visibleWidth()` from `@earendil-works/pi-tui`, not `.length`. Emojis (🟢⚪⬛📊▶) are 2 terminal columns. Using `.length` causes border overflow/glitch.
- **Overlay**: `render(width)` must use the `width` param from Pi for the border. Fixed widths cause clipping on narrow terminals.
- **Overlay options**: use `width: "80%"`, `maxWidth`, `maxHeight: "90%"` for responsive sizing.
- **Router mode**: `/props` returns router-level info only. Status and context size come from `/models` `status.args`. Slots/metrics need `?model=X` query param.
- **Metrics requires `--metrics`**: `/metrics` returns 501 if server started without `--metrics` flag. Gracefully degrades (shows nothing).
- **Slots may be disabled**: `/slots` can be disabled with `--no-slots`. Gracefully degrades.
- **Context size fallback**: models with no `--ctx-size` in args get 32768 default.
- **Remote is opt-in**: no default remote URL. Must be set explicitly in `settings.json`.
- **Provider IDs**: both `llama-server` and `llama-cpp` are accepted for unload checks.
- **Multi-server**: `rpc` takes a `ServerConfig`, not a global URL. All helpers are per-server.
- **V1 models ID matching**: `/v1/models` may return full paths; match by checking if ID ends with the model id from `/models`.
- **Metrics parsing**: Prometheus text format — skip `#` comments, split on last space for value.
- **Load UI**: `/llama-load` with no args shows `ctx.ui.select` picker. With an arg, loads directly.
