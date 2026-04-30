# justembed Daemon Architecture

**Audience**: Developers, operators

## Overview

`justembed` is a gRPC daemon that loads a single GGUF embedding model (currently nomic-bert) and exposes `justembed.EmbedService/Embed` over TCP (default port 8181). Input texts are tokenized, passed through the model with mean pooling, and L2-normalized before returning. The result is a batch of unit-length float vectors ready for cosine-similarity search.

The daemon is the composition root: it loads one model at startup and serves it until it exits. Hot-swapping, multi-model, and multi-tenant are out of scope — one model, one process, one port.

## Crate Structure

```
justembed/
├── conversion/          (swe-ml-embedding — embedding primitives, no I/O, no async)
│   └── src/
│       ├── api/         # Public traits, types, errors
│       │   ├── traits.rs    # Embed, Normalize
│       │   ├── types.rs     # EmbeddingsRequest, EmbeddingsResponse (OpenAI-compat shape)
│       │   └── error.rs     # EmbeddingError, EmbeddingResult<T>
│       ├── core/        # Implementations (private to crate)
│       │   ├── embedding.rs # DefaultEmbedding (impl Embed)
│       │   └── normalize.rs # L2Normalize (impl Normalize)
│       └── saf/         # Curated re-exports
│
└── systemd/             (swe-embedding-systemd — gRPC daemon)
    └── src/
        ├── api/         # Config + proto re-exports
        │   ├── config.rs    # AppConfig, EmbeddingGrpcConfig, TLS config
        │   └── proto.rs     # include! of generated proto types
        ├── core/        # Implementations (private to crate)
        │   ├── state.rs     # EmbeddingState: model + tokenizer + model-id
        │   ├── embed.rs     # embed_inputs() — the main embedding loop
        │   ├── loader.rs    # load_gguf() — GGUF file → EmbeddingState
        │   ├── grpc_handler.rs  # EmbedHandler (impl Handler)
        │   ├── grpc_server.rs   # start_grpc_server() — wires tonic + routing
        │   └── grpc_dispatch.rs # MethodPathRouter — routes by URI prefix
        ├── saf/         # Curated re-exports
        └── bin/
            └── embed.rs # CLI entry point + composition root
```

Both crates follow the SEA layering: interfaces in `api/`, implementations in `core/`, public facade in `saf/`. Nothing in `core/` is `pub` outside the crate — only `saf/mod.rs` decides what crosses the boundary.

## Key Design Decisions

### Program to interfaces: `Embed` + `Normalize`

Two traits mediate between the embedding loop and the things it depends on:

- **`Embed`** (`conversion/src/api/traits.rs`) — `forward(indices: &Tensor) -> EmbeddingResult<Tensor>`, `num_embeddings()`, `embedding_dim()`. Implemented by `DefaultEmbedding`. `embed_inputs` never touches the weight matrix directly.
- **`Normalize`** (`conversion/src/api/traits.rs`) — `normalize(v: &mut [f32]) -> EmbeddingResult<()>`. Implemented by `L2Normalize`. The normalization step is swappable without touching the embedding loop.

On the gRPC side, `EmbedHandler` implements the upstream `Handler<Vec<u8>, Vec<u8>>` trait. `HandlerRegistryDispatcher` implements `GrpcInbound`. The server wires concrete types once in `grpc_server.rs`; nothing downstream knows about them.

### No-model mode (graceful degradation)

If `[embedding.model].gguf_path` is empty or absent, the daemon boots successfully and binds the gRPC port. Every `Embed` RPC returns `FailedPrecondition` (status 9). The health service reports `NOT_SERVING`.

This matters because the daemon is often deployed before its model file is mounted. Returning a clean gRPC status gives callers a retry signal; "connection refused" gives them nothing. The model path is checked once at startup in `bin/embed.rs` — there is no live-reload.

### gRPC routing: `MethodPathRouter` + `HandlerRegistryDispatcher`

All inbound requests arrive at a single tonic service. `MethodPathRouter` (in `grpc_dispatch.rs`) routes by URI prefix:

- `/grpc.health.v1.Health/*` → `HealthService` (standard health protocol)
- everything else → `HandlerRegistryDispatcher`

`HandlerRegistryDispatcher` looks up the handler by full method path. Today exactly one handler is registered (`EmbedHandler`, pattern `/justembed.EmbedService/Embed`). Adding a new RPC method means implementing `Handler`, registering it, and updating the proto — no match statements, no dispatch table in application code.

The health service runs a refresh task every 2 seconds that polls each registered handler's `health_check()`. The aggregate status is `SERVING` only when all handlers report healthy (model loaded). This means no-model mode is visible to health checks automatically.

### Request handling: `spawn_blocking` for embedding

Tokenization and the model forward pass are CPU-bound and would starve the tokio async runtime if run directly in handler futures. `EmbedHandler::execute()` calls `tokio::task::spawn_blocking` to move `embed_inputs()` onto the blocking thread pool.

Unlike `swellmd` (which has explicit admission control via a semaphore), `justembed` relies on the upstream edge ingress deadline (default 30 s) and OS-level scheduling. If concurrent embedding requests become a problem (thread pool saturation), a `Throttle` wrapper following the `swellmd` pattern is the right fix — add it to the `Handler` trait wrapper, not inside `embed_inputs`.

### `embed_inputs` pipeline

`embed_inputs(state, inputs)` in `systemd/src/core/embed.rs` is the single function that performs the full embedding work:

```
1. Reject empty input (EmbedError::EmptyInput → InvalidArgument)
2. For each text: tokenize → Vec<u32>
3. For each token sequence:
   a. Build Tensor [1, seq_len]
   b. model.embed(tensor, PoolingStrategy::Mean) → Tensor [1, dim]
   c. L2Normalize the output vector in-place
4. Return Vec<Vec<f32>> + total token count
```

Errors at each stage surface as distinct variants (`Tokenization`, `Embed`, `Normalize`) so the gRPC handler can map them to the right status code. `Tokenization` is `InvalidArgument` (client-fixable); `Embed`/`Normalize` are `Internal` (sanitized on wire).

### Error model

`EmbedError` → gRPC status mapping in `EmbedHandler::execute()`:

| Variant | gRPC status | Wire message |
|---------|-------------|-------------|
| `EmptyInput` | `InvalidArgument` (3) | verbatim |
| `Tokenization(msg)` | `InvalidArgument` (3) | verbatim |
| `Embed(msg)` | `Internal` (13) | sanitized constant |
| `Normalize(msg)` | `Internal` (13) | sanitized constant |
| Model not loaded | `FailedPrecondition` (9) | fixed string |

The edge ingress layer handles the `Internal` sanitization — the raw message is logged WARN server-side only (see T4 in `threat_model.md`).

## gRPC Methods

| Method | Proto | Purpose |
|--------|-------|---------|
| `Embed` | `justembed.EmbedService/Embed` | Batch text → L2-normalized float vectors |
| `Check` | `grpc.health.v1.Health/Check` | Liveness + serving status |
| `Watch` | `grpc.health.v1.Health/Watch` | Streaming health updates |

Request/response types are in `systemd/proto/embed.proto`. Proto types are generated at compile time by `systemd/build.rs` via `tonic-build`.

## Lifecycle

1. `bin/embed.rs` parses the CLI (`embed serve`) via clap.
2. Config is loaded via the `swe-systemd` XDG loader: bundled `application.toml` → `$XDG_CONFIG_DIRS/llminference/application.toml` → `$XDG_CONFIG_HOME/llminference/application.toml`. Later layers deep-merge over earlier ones.
3. `RUST_LOG` is configured from `[logging].level`.
4. If `gguf_path` is non-empty: `load_gguf(path)` parses weights + tokenizer into `Arc<EmbeddingState>`. If empty: `Arc<EmbeddingState>` is constructed without a model (no-model mode).
5. `EmbedHandler` is constructed with `Arc<EmbeddingState>` and registered into a `HandlerRegistry`.
6. `start_grpc_server(config, registry)` builds `MethodPathRouter` (health + dispatcher), wraps it in `TonicGrpcServer`, and binds to `[embedding.grpc].host:[embedding.grpc].port`.
7. The server task is awaited until SIGINT. No graceful drain — the process is the deployment unit.

## What's Out of Scope

- **Multi-model serving**: one process, one model. Run multiple daemons on different ports.
- **Streaming embeddings**: the gRPC method is unary. Batch inputs in a single request.
- **Authentication**: not wired (see T6 in `threat_model.md`). Network policy or mTLS client cert required for production.
- **Dynamic model reload**: model path is read once at startup. Reload = process restart.
- **GPU**: CPU-only via the `swe-llmmodel-*` crates. No CUDA/Vulkan path in this daemon.
- **Quantization**: the loaded GGUF is used as-is. Post-load quantization is a future concern.

## Testing

- Unit tests are co-located with their modules. Run with `cargo test --workspace`.
- Integration tests in `systemd/tests/` spin up a real in-process gRPC server on a random port — no mocks.
  - `grpc_embed_int_test.rs`: full proto round-trip, dispatch, and error-code mapping.
  - `grpc_health_int_test.rs`: `Check` and `Watch` against the health service.
- For throughput and latency numbers see `../7-operations/benchmarks/baseline.md`.
