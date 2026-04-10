# RustML Developer Guide

> **TLDR:** Architecture overview and contributor guide for the RustML workspace — crate structure, SEA layering, build system, testing, and the swellmd daemon.

**Audience**: Developers contributing to or extending the RustML codebase

**WHAT**: Complete developer reference for the RustML workspace
**WHY**: Onboard new contributors and maintain architectural consistency across crates
**HOW**: Architecture diagrams, crate descriptions, coding conventions, and development workflows

---

## Table of Contents

- [Workspace Overview](#workspace-overview)
- [Crate Architecture](#crate-architecture)
  - [rustml-core](#rustml-core)
  - [rustml-nn](#rustml-nn)
  - [rustml-hub](#rustml-hub)
  - [rustml-nlp](#rustml-nlp)
  - [rustml-tokenizer](#rustml-tokenizer)
  - [rustml-quant](#rustml-quant)
  - [rustml-gguf](#rustml-gguf)
  - [rustml-swets](#rustml-swets)
  - [rustml-cli](#rustml-cli)
  - [swellmd (daemon)](#swellmd-daemon)
- [SEA Layering Pattern](#sea-layering-pattern)
- [Dependency Graph](#dependency-graph)
- [Development Workflow](#development-workflow)
  - [Building](#building)
  - [Testing](#testing)
  - [Adding a New Crate](#adding-a-new-crate)
- [Key Abstractions](#key-abstractions)
  - [Tensor](#tensor)
  - [LanguageModel Trait](#languagemodel-trait)
  - [Tokenizer Trait](#tokenizer-trait)
  - [Generator](#generator)
- [SwellMD Daemon Architecture](#swellmd-daemon-architecture)
  - [Request Flow](#request-flow)
  - [State Management](#state-management)
  - [Streaming Design](#streaming-design)
- [Coding Conventions](#coding-conventions)
- [Performance Guidelines](#performance-guidelines)
- [See Also](#see-also)

---

## Workspace Overview

RustML is a Cargo workspace rooted at `machinelearning/`. Two workspaces: `llm/` for LLM inference and `ml-sdk/` for general ML:

```
machinelearning/
├── Cargo.toml              # Workspace root — members, shared deps, profiles
├── llm/                    # LLM inference stack
│   ├── core/               # rustml-core — tensors, SIMD ops, arena allocator
│   ├── nn/                 # rustml-nn — transformer layers, attention, KV cache
│   ├── hub/                # rustml-hub — HuggingFace download + SafeTensors
│   ├── nlp/                # rustml-nlp — LLM models, generation, sampling
│   ├── tokenizer/          # rustml-tokenizer — BPE, HF, GGUF tokenizers
│   ├── quant/              # rustml-quant — Q4_0/Q4_1/Q8_0 quantization + SIMD
│   ├── gguf/               # rustml-gguf — GGUF binary format parser
│   ├── quantize/           # rustml-quantize — SafeTensors to GGUF pipeline
│   ├── cli/                # rustml-cli — unified `sweai` binary
│   └── daemon/             # swellmd — HTTP inference daemon
├── ml-sdk/                 # ML SDK
│   └── swets/              # rustml-swets — time series training (experimental)
└── docs/                   # Project documentation
```

**Edition**: Rust 2024 | **Profile**: thin LTO in release | **License**: MIT OR Apache-2.0

---

## Crate Architecture

### rustml-core

Foundational tensor operations and runtime configuration.

- `Tensor` type with F32, F16, Q4_0, Q4_1, Q8_0 dtypes
- SIMD-accelerated ops: matmul, softmax, layer norm, RMS norm (AVX2/SSE4.1/NEON)
- Arena allocator for batch memory management
- `RuntimeConfig` with atomic parallelization thresholds
- `OptProfile` enum: Optimized, Aggressive, Baseline

### rustml-nn

Transformer building blocks.

- `MultiHeadAttention` — causal/non-causal, GQA, RoPE, QK normalization
- `TransformerBlock` — full layer with residual connections
- `FeedForward` — standard FFN and gated variants (SwiGLU, GeGLU)
- `Linear` — quantization-aware matrix multiplication
- `KVCache` — pre-allocated cache for autoregressive O(n) decode
- `RoPE` — rotary position encoding
- `MoeLayer` — mixture of experts (Mixtral)

### rustml-hub

HuggingFace Hub integration.

- `HubApi` — download models with token authentication
- Token resolution: `--token` flag > `HF_TOKEN` env > `~/.cache/huggingface/token`
- SafeTensors format parsing
- Per-architecture weight mapping (GPT-2, Llama, Gemma, Falcon, etc.)
- Cache at `~/.cache/rustml/hub/`

### rustml-nlp

LLM implementations and text generation.

- `LlmModel` — unified model struct supporting GPT-2, Llama, Gemma 3, Falcon, Mixtral, Nomic-BERT
- `GptModel` — reference/teaching implementation (O(n^2) per-token, no KV cache)
- `Generator` — autoregressive generation with streaming, multi-turn chat, batch
- Sampling: temperature, top-k, top-p, repetition penalty
- `ModelConfig` — architecture dispatch via configuration (no if-let spaghetti)

### rustml-tokenizer

Text tokenization with multiple backends.

- `Tokenizer` trait — encode/decode/vocab_size/token_to_id
- `BpeTokenizer` — GPT-2 style byte-pair encoding
- `HFTokenizer` — HuggingFace `tokenizers` crate wrapper
- `GgufTokenizer` — extracted from GGUF metadata
- `ByteTokenizer` — simple byte-level encoding

### rustml-quant

Quantization engine with SIMD kernels.

- Q8_0: 1 f16 scale + 32 i8 values per block
- Q4_0: 1 f16 scale + 16 nibble values per block
- Q4_1: f16 scale + f16 min + 16 nibble values per block
- SIMD dot product kernels for quantized matmul
- Auto threshold: dimensions >= 768 -> Q8_0

### rustml-gguf

GGUF binary format parser (llama.cpp ecosystem).

- Header/metadata/tensor parsing with alignment handling
- Architecture detection (llama, gemma3, falcon, mixtral, nomic-bert)
- Weight remapping to unified LlmModel format
- Memory-mapped file I/O via `memmap2`

### rustml-swets

Time series training framework (experimental).

- Autodiff engine with backpropagation
- LSTM, Conv1D, BatchNorm1d layers
- Optimizers: Adam, SGD
- Losses: MSE, L1, QuantileLoss
- Trainer builder pattern

### rustml-cli

Unified `sweai` CLI binary.

- Subcommands: `infer`, `hub`, `gguf`, `tokenizer`
- Interactive chat mode
- Batch generation
- See [User Manual](../7-operations/user_manual.md)

### swellmd (daemon)

HTTP inference daemon serving OpenAI-compatible API.

- Axum-based HTTP server
- Endpoints: `/health`, `/v1/models`, `/v1/chat/completions`
- Blocking and SSE streaming modes
- Single-model server with `Arc<AppState>` shared state
- See [SwellMD Daemon Architecture](#swellmd-daemon-architecture) below

---

## SEA Layering Pattern

Every crate follows **Stratified Encapsulation Architecture** (SEA) with three layers:

```
my_crate/
├── Cargo.toml
├── main/src/
│   ├── lib.rs          # pub mod api; mod core; mod saf; pub use saf::*;
│   ├── api/            # Public contracts: types, traits, errors
│   │   ├── mod.rs
│   │   ├── types.rs    # Public structs, enums, configs
│   │   └── error.rs    # Error types with thiserror
│   ├── core/           # Implementation (private to crate)
│   │   ├── mod.rs
│   │   └── ...         # Business logic
│   ├── saf/            # Facade re-exports
│   │   └── mod.rs      # pub use crate::api::*; pub use crate::core::SomePublicThing;
│   └── spi/            # Service Provider Interface (optional)
│       └── contract/   # Traits for pluggable implementations
└── tests/              # Integration tests
```

**Rules:**
- `api/` defines public contracts — types, traits, errors
- `core/` contains implementation — private, only exposed via `saf/`
- `saf/` is the facade — curated re-exports forming the public API
- `spi/` defines extension points (optional) — traits for pluggable backends
- `lib.rs` wires it together: `pub mod api; pub(crate) mod core; mod saf; pub use saf::*;`

---

## Dependency Graph

```
                    rustml-core
                   /     |      \
            rustml-nn  rustml-quant  rustml-hub
              |    \      |          /
           rustml-nlp  rustml-gguf  rustml-tokenizer
              |         /        /
           rustml-cli  /        /
              |       /        /
           swellmd  ----------
```

Key: arrows point from dependent to dependency.

---

## Development Workflow

### Building

```bash
# Full workspace (all crates)
cargo build

# Single crate
cargo build -p swellmd

# Release (optimized, thin LTO)
cargo build -p swellmd --release

# Check without building (fast feedback)
cargo check -p swellmd
```

### Testing

```bash
# All tests (may require models for some integration tests)
cargo test

# Single crate unit tests
cargo test -p rustml-core

# Single integration test
cargo test -p rustml-nlp --test forward_diagnostic_test

# With logging
RUST_LOG=rustml=debug cargo test -p rustml-nlp -- --nocapture
```

**Note:** Some integration tests in `rustml-nlp` require downloaded models. These are gated behind `#[ignore]` or require manual setup. See [Model Verification Guide](guides/model-verification.md).

### Adding a New Crate

1. Create the directory structure following SEA:
   ```bash
   mkdir -p llm/my_crate/main/src/{api,core,saf}
   mkdir -p llm/my_crate/tests
   ```

2. Create `Cargo.toml` using workspace inheritance:
   ```toml
   [package]
   name = "rustml-my-crate"
   version.workspace = true
   edition.workspace = true
   license.workspace = true
   ```

3. Add to workspace `Cargo.toml`:
   ```toml
   [workspace]
   members = [
       # ...
       "llm/my_crate",
   ]

   [workspace.dependencies]
   rustml-my-crate = { path = "llm/my_crate" }
   ```

4. Create `lib.rs`:
   ```rust
   pub mod api;
   pub(crate) mod core;
   mod saf;
   pub use saf::*;
   ```

5. Note: `gen` is a reserved keyword in Rust 2024 edition. Use `generator`, `produce`, or another name instead.

---

## Key Abstractions

### Tensor

The fundamental data type (`rustml-core`):

```rust
pub struct Tensor {
    data: Vec<u8>,       // Raw bytes (dtype-dependent)
    shape: Shape,        // [B, S, D] or [rows, cols]
    dtype: DType,        // F32, F16, Q4_0, Q4_1, Q8_0
}
```

### LanguageModel Trait

Defines the interface for all LLMs (`rustml-nlp`):

```rust
pub trait LanguageModel {
    fn forward(&self, input_ids: &Tensor) -> NlpResult<Tensor>;
    fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KVCache) -> NlpResult<Tensor>;
    fn vocab_size(&self) -> usize;
    fn max_sequence_length(&self) -> usize;
    fn embedding_dim(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
}
```

Both `GptModel` and `LlmModel` implement this trait.

### Tokenizer Trait

Pluggable tokenization (`rustml-tokenizer`):

```rust
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError>;
    fn vocab_size(&self) -> usize;
    fn token_to_id(&self, token: &str) -> Option<u32>;
}
```

### Generator

High-level inference engine (`rustml-nlp`):

```rust
let generator = Generator::new(model, tokenizer, temperature)
    .with_top_k(40)
    .with_top_p(0.9)
    .with_eos_token(eos_id)
    .with_chat_template(template)
    .with_context_len(2048);

// Single prompt
let output = generator.generate("Hello", 256)?;

// Streaming
generator.generate_stream("Hello", 256, |token_id| {
    print!("{}", tokenizer.decode(&[token_id]).unwrap());
    true // continue
})?;

// Multi-turn chat
let messages = vec![("user", "What is Rust?"), ("assistant", "A systems language.")];
generator.generate_turn_stream(&messages, 256, |token_id| { ... })?;
```

---

## SwellMD Daemon Architecture

### Crate Structure

```
llm/daemon/
├── Cargo.toml
├── main/src/
│   ├── lib.rs
│   ├── api/
│   │   ├── error.rs        # DaemonError — maps to HTTP status codes
│   │   └── types.rs        # OpenAI-compatible request/response types
│   ├── core/
│   │   ├── loader.rs       # load_gguf() / load_safetensors()
│   │   ├── router.rs       # Axum HTTP handlers
│   │   └── state.rs        # ModelBundle + AppState
│   ├── saf/
│   │   └── mod.rs          # Facade re-exports
│   └── bin/
│       └── serve.rs        # swellmd binary entry point
└── tests/
```

### Request Flow

```
HTTP Request
    │
    ▼
axum::Router (async)
    │
    ├─ GET /health           → health()          → JSON response
    ├─ GET /v1/models        → list_models()     → JSON response
    └─ POST /v1/chat/completions
        │
        ▼
    validate_request()
        │
        ├─ stream=false → handle_blocking()
        │                    │
        │                    ▼
        │               spawn_blocking {
        │                   ModelBundle::build_generator()
        │                   generator.generate_turn_stream()
        │               }
        │                    │
        │                    ▼
        │               ChatCompletionResponse (JSON)
        │
        └─ stream=true  → handle_streaming()
                             │
                             ▼
                        spawn_blocking {
                            generator.generate_turn_stream(callback)
                            callback sends tokens via mpsc channel
                        }
                             │
                             ▼
                        ReceiverStream → SSE events
                        "data: {chunk}\n\n" per token
                        "data: [DONE]\n\n" at end
```

### State Management

```rust
// Shared across all requests via Arc
pub struct AppState {
    pub bundle: ModelBundle,
}

pub struct ModelBundle {
    pub model: LlmModel,                         // Loaded model weights
    pub tokenizer: Box<dyn Tokenizer + Send + Sync>, // Tokenizer
    pub model_id: String,                         // Display name
    pub chat_template: Option<String>,            // Chat format template
    pub eos_token_id: Option<u32>,                // End-of-sequence token
    pub bos_token_id: Option<u32>,                // Beginning-of-sequence token
    pub profile: OptProfile,                      // Performance profile
}
```

The model is loaded once at startup and shared read-only across requests. `Generator` is built per-request (cheap — just borrows the model and tokenizer).

### Streaming Design

SSE streaming uses a `tokio::sync::mpsc` channel to bridge the blocking inference thread and the async response stream:

1. `handle_streaming()` creates an `mpsc::channel(64)`
2. `spawn_blocking` runs inference, sending decoded tokens through the channel
3. The async side wraps the receiver as `ReceiverStream`, maps each piece to an SSE `Event`
4. A `[DONE]` sentinel is appended after the token stream ends
5. Axum's `Sse` adapter sends the events with keep-alive

---

## Coding Conventions

### Naming

- **Types**: PascalCase nouns (`ModelBundle`, `DaemonError`, `ChatCompletionRequest`)
- **Functions**: snake_case verbs (`load_gguf`, `build_router`, `validate_request`)
- **Constants**: SCREAMING_SNAKE (`GPT2_EOS_TOKEN_ID`)
- **Tests**: `test_<action>_<condition>_<expectation>`
- **Avoid** `gen` as a variable name (reserved keyword in Rust 2024)

### Error Handling

- Public APIs return `Result<T, CrateError>` using `thiserror`
- Each crate defines its own error enum in `api/error.rs`
- Use `anyhow` only in binaries, not libraries
- Map errors at crate boundaries with `From` impls

### Module Layout

- Follow SEA: `api/` (public), `core/` (private), `saf/` (facade)
- One responsibility per file
- Re-export the curated public API through `saf/mod.rs`
- Keep `lib.rs` minimal: just module declarations and `pub use saf::*`

---

## Performance Guidelines

- **Release builds**: Always use `--release` for inference. Debug builds are 5-10x slower.
- **Quantization**: SafeTensors models auto-quantize F32 -> Q8_0 at load time (75% memory reduction, 10% faster).
- **Weight fusion**: `fuse_gate_up_weights()` and `fuse_qkv_weights()` reduce dispatch overhead.
- **Warmup**: `model.warmup_decode()` primes rayon, SIMD, and branch prediction. Called automatically in both CLI and daemon loaders.
- **Profiling**: Use `RUST_LOG=rustml=trace` for per-layer timing breakdown.
- **Blocking threads**: All inference in the daemon runs on `spawn_blocking` to keep the async runtime responsive.

---

## See Also

- [Operations Guide](../7-operations/operations_guide.md) — Running swellmd in production
- [Deployment Guide](../6-deployment/deployment_guide.md) — Building and deploying
- [User Manual](../7-operations/user_manual.md) — `sweai` CLI reference
- [Model Verification Guide](guides/model-verification.md) — Verifying model correctness
- [Backlog](backlog.md) — Open work items and performance notes
- [ADR-001: Unified LlmModel](../3-design/adr/adr-001-unified-llmmodel-for-gpt2.md) — Architecture decision
