# swellmd Daemon Architecture

**Audience**: Developers, operators

## Overview

`swellmd` is an HTTP daemon that serves a loaded LLM over an OpenAI-compatible API. It wraps a `rustml-model::LlmModel`, a tokenizer, and generation config into a single process that handles `chat/completions` (blocking + streaming), `embeddings`, model metadata, and health checks.

The daemon is the composition root: it loads one model at startup, applies a fixed optimization profile, and serves that model until it exits. Hot-swapping, multi-tenancy, and distributed coordination are out of scope — this is one model, one process, one port.

## Crate Structure

```
daemon/main/src/
├── api/               # Public traits and types (no logic)
│   ├── error.rs       # DaemonError enum + HTTP status mapping
│   ├── model.rs       # Model trait — runtime-ready model interface
│   ├── throttle.rs    # Throttle trait + Permit (admission control)
│   └── types.rs       # OpenAI-compatible request/response DTOs
├── core/              # Implementations
│   ├── loader.rs      # load_gguf(), load_safetensors() — build DefaultModel
│   ├── router.rs      # axum routes + handler bodies
│   ├── state.rs       # DefaultModel (impl Model) + AppState
│   └── throttle.rs    # SemaphoreThrottle (impl Throttle)
├── saf/               # Facade re-exports
│   └── mod.rs
├── bin/
│   └── serve.rs       # CLI entry point + composition root
└── lib.rs
```

The daemon follows the standard SEA layering: interfaces in `api/`, implementations in `core/`, public facade in `saf/`. Consumers depend on traits; concrete types are constructed once in `bin/serve.rs` and injected.

## Key Design Decisions

### Program to interfaces: `Model` + `Throttle`

Two traits mediate between handlers and the things they depend on:

- **`Model`** (`api/model.rs`) — what a handler needs from a loaded model: `model_id()`, `build_generator(temperature)`, `tokenizer()`, `embed(input, strategy)`. The router never touches `LlmModel` directly.
- **`Throttle`** (`api/throttle.rs`) — admission control: `try_acquire()`, `capacity()`, `available()`. Returns an RAII `Permit` whose drop releases the slot.

`DefaultModel` (holds `LlmModel` + tokenizer + generation config) implements `Model`. `SemaphoreThrottle` (wraps `tokio::sync::Semaphore` — see [glossary](../../../../../../docs/3-design/glossary.md#semaphore)) implements `Throttle`. `AppState` stores `Box<dyn Model>` and `Box<dyn Throttle>` — the router is fully abstract over implementation.

This matters because:
- Tests can substitute fakes without loading a real model.
- Alternative throttles (distributed limiter, no-op for dev) drop in without handler changes.
- The name collisions we had before (two `ModelBundle` types in hub/ and daemon/) are gone — the daemon owns its own abstraction.

### Model loading: architecture-dispatch via `ModelBuilderRegistry`

`core/loader.rs` builds a `ModelBuilderRegistry` from `rustml-model` that maps architecture strings → per-architecture builders (`LlamaBuilder`, `Gpt2Builder`, `Gemma3Builder`, etc.). The loader reads `config.architecture` from the model's config.json (SafeTensors) or GGUF metadata, then hands the registry the weights and config. **No match statements on architecture strings in the daemon** — dispatch is config-driven.

Loading paths:
- `load_gguf(path, profile)` — mmap-based zero-copy load, tokenizer extracted from GGUF metadata
- `load_safetensors(model_id, profile)` — HuggingFace Hub download (cached), mmap-based load via `rustml_hub::load_safetensors`, tokenizer from `tokenizer.json` or BPE vocab/merges

After weights are loaded, the loader runs post-construction optimization: config-driven quantization (per layer type, from `quantization.toml`), then weight fusion (gate+up projection, QKV projection). A single forward-pass warmup triggers any lazy initialization so the first real request isn't slow.

### Admission control: fail fast, don't queue

Each compute handler (`handle_blocking`, `handle_streaming`, `embeddings`) calls `state.throttle.try_acquire()` before `tokio::task::spawn_blocking`. On `None`, the handler returns `DaemonError::AtCapacity(capacity)` → HTTP 503 with body `{"error":{"type":"at_capacity","message":"Server at capacity (max concurrent=N)"}}`.

The permit moves into the `spawn_blocking` closure and drops when inference completes, releasing the slot.

Rationale for **fail fast over queueing**: CPU inference is slow (seconds per request). Silently queueing a 50-request burst behind a 2-wide throttle means the last client waits 25× the single-request latency with no indication. Returning 503 immediately gives clients a retry signal and keeps tail latency bounded. If async admission is needed later, `acquire().await` with a bounded timeout can be added to the trait without breaking callers.

Default capacity is `[throttle.semaphore].max_concurrent = 2` in `application.toml`. Higher values oversubscribe the rayon thread pool inside each request. For a 1B model on 8 cores, empirically 2–4 is the sweet spot (see `docs/5-testing/report/load_testing_2026_04_12_admission_control.md`).

### Request handling: `spawn_blocking` for compute

Generation and embedding are CPU-bound and would starve tokio's async runtime if run directly in handler futures. All compute runs on the blocking thread pool via `tokio::task::spawn_blocking`. The blocking pool is unbounded by default (512 threads), which is why the explicit `Throttle` is required — without it, the bound becomes "until the allocator fails."

Streaming uses a `tokio::sync::mpsc::channel` to ferry tokens from the blocking worker back to the SSE response stream. The blocking closure decodes token IDs to pieces via the tokenizer and calls `tx.blocking_send(piece)`. The async side wraps the receiver as a `ReceiverStream`, maps pieces to SSE `Event`s, and appends a final `[DONE]` terminator.

### Error model

`DaemonError` (`api/error.rs`) is the single error type handlers return. It implements `IntoResponse` with a status-code mapping:

| Variant | HTTP | Type field |
|---|---|---|
| `InvalidRequest(msg)` | 400 | `invalid_request_error` |
| `GenerationFailed(msg)` | 500 | `generation_error` |
| `LoadFailed(msg)` | 500 | `load_error` |
| `Internal(msg)` | 500 | `internal_error` |
| `ModelNotLoaded(msg)` | 503 | `model_not_loaded` |
| `AtCapacity(N)` | 503 | `at_capacity` |

Error body shape matches OpenAI: `{"error":{"message":..., "type":...}}`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | Liveness + loaded model id |
| GET  | `/v1/models` | Single-entry model list (OpenAI format) |
| POST | `/v1/chat/completions` | Chat, optional `stream: true` for SSE |
| POST | `/v1/embeddings` | Batch embeddings with L2-normalized output |

Request/response DTOs live in `api/types.rs` and mirror OpenAI's field names so existing client SDKs (openai-python, openai-node, curl scripts) work without changes.

## Lifecycle

1. `bin/serve.rs` calls `load_config()` which reads `llmserv/main/config/application.toml` (bundled via `include_str!`) and overlays any user config from `$XDG_CONFIG_DIRS/llmserv/application.toml` and `$XDG_CONFIG_HOME/llmserv/application.toml` via deep-merge. The merged result deserializes into a typed `AppConfig`.
2. `RUST_LOG` is set from `[logging].level` if not already in the environment.
3. The optimization profile (`[runtime].opt_profile`) is applied to the `rustml-tensor` runtime config.
4. `rayon` thread pool size is logged (from `rustml-thread-config::AutoThreadConfig`).
5. The model is loaded via `load_gguf` (when `[model].source = "gguf"`) or `load_safetensors` (when `[model].source = "safetensors"`). The merged TOML is passed through so the runtime quantizer can read its `[quantization]` section without re-reading files.
6. `build_throttle(cfg)` dispatches on `[throttle].provider` to construct the admission-control backend. Today only `"semaphore"` is implemented; new providers slot in as one `match` arm plus a config sub-table.
7. `AppState { model, throttle }` is wrapped in `Arc` and handed to `build_router`.
8. axum binds to `[server].host:[server].port` and serves until SIGINT/SIGTERM.

No CLI flags. No reload. No graceful shutdown drain — the process is the unit of deployment. To run with different settings, set `XDG_CONFIG_HOME=/path/to/dir` where the overlay lives at `$XDG_CONFIG_HOME/llmserv/application.toml`.

## What's Out of Scope

- **Multi-model serving**: one process, one model. Run multiple daemons on different ports.
- **GPU**: CPU-only. See `BACKLOG.md` P6 for the Vulkan plan in `rustml-compute`.
- **Authentication**: bearer tokens, API keys, quota — all expected to sit in a reverse proxy (nginx, Caddy, Envoy).
- **Persistent queueing**: no durable buffer between client and handler. Failed-fast 503 is the backpressure signal; clients retry.
- **Continuous batching**: requests run sequentially through `spawn_blocking`. Static batching across requests requires the generation layer to support it — not wired yet.

## Testing

- Unit tests for `SemaphoreThrottle` live in `core/throttle.rs` (capacity, permit release, clamp-to-1).
- End-to-end load tests via `scripts/load_test.sh N [URL]` (relative to this crate). See `docs/5-testing/report/load_testing_2026_04_12_admission_control.md` for the burst/capacity matrix.
- HTTP integration tests are absent — the OpenAI-compat shape is validated manually against `curl` and the openai-python client.
