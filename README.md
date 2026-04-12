# machinelearning

Pure-Rust ML platform. Two Cargo workspaces plus a developer-tools tree.

## Layout

```
machinelearning/
├── main/features/          — foundation crates (workspace root at Cargo.toml)
│   ├── tensor/             swe-ml-tensor — multi-dtype tensors, SIMD, mmap storage
│   ├── activation/         swe-ml-activation
│   ├── normalization/      swe-ml-normalization
│   ├── architectures/      swe-ml-architectures — shared model architecture defs
│   ├── hub/                rustml-hub — HuggingFace Hub client + SafeTensors loader
│   ├── embedding/conversion/   swe-ml-embedding — embedding utilities
│   └── training/           swe-ml-training — training primitives
│
├── llmserv/                — LLM serving workspace (own Cargo.toml)
│   ├── main/config/application.toml    single source of truth for all llmserv apps
│   └── main/features/
│       ├── inference/      the compute library — model, layers, architectures,
│       │                   compute, generation, gguf, prefill, quant, quantizer,
│       │                   thread-config, tokenizer
│       ├── daemon/         swellmd — HTTP chat-completions server (leaf frontend)
│       ├── cli/            sweai — multi-command developer CLI
│       ├── embedding/server/     swe-ml-embed — embedding HTTP server
│       └── experimentation/llmforge/   archived prototype (read-only)
│
└── devtools/
    └── quantize/           rustml-quantize — offline SafeTensors → GGUF CLI tool
```

Cross-workspace dependency rule: **llmserv → main (foundations)**. Root workspace
never depends on llmserv, with one exception: `devtools/quantize` reaches into llmserv
for its quantizer library. This is the single backward arrow, accepted because quantize
is a tool, not a library.

## Quick start

### Build everything

```bash
# Root workspace (foundations + devtools)
cargo build --release

# llmserv workspace (inference stack, daemon, cli, embedding server)
cargo build --release --manifest-path llmserv/Cargo.toml
```

### Run the LLM daemon

swellmd is config-driven. Edit `llmserv/main/config/application.toml` to pick the model,
or set an override at `$XDG_CONFIG_HOME/llmserv/application.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8080

[model]
source = "safetensors"                  # or "gguf"
id = "google/gemma-3-1b-it"             # HuggingFace repo ID
# path = "./models/gemma-3-1b-it-Q4_0.gguf"   # when source = "gguf"
```

Then:

```bash
llmserv/target/release/swellmd
# → serves OpenAI-compatible /v1/chat/completions on configured host/port
```

For gated HuggingFace models, set `HF_TOKEN=hf_...` in the environment.

### Quantize a model (optional, offline)

```bash
cargo run --release -p rustml-quantize -- \
  --model google/gemma-3-1b-it \
  --target q4_0 \
  --output ./models/gemma-3-1b-it-q4_0.gguf
```

Point the daemon at the resulting GGUF (`[model].source = "gguf"`) for smaller on-disk
size and faster startup. See `devtools/quantize/README.md` for details.

### Embedding server

```toml
[embedding.server]
host = "127.0.0.1"
port = 8081

[embedding.model]
gguf_path = "./models/nomic-embed-text-v1.5.Q8_0.gguf"
```

```bash
llmserv/target/release/swe-ml-embed
# → serves OpenAI-compatible /v1/embeddings on configured host/port
```

### Developer CLI (sweai)

Unlike the daemons, `sweai` is flag-driven (single-invocation dev tool):

```bash
llmserv/target/release/sweai --help
```

Subcommands: `infer`, `hub`, `gguf`, `tokenizer`.

## Documentation

| Topic | Location |
|---|---|
| Daemon architecture | `llmserv/main/features/daemon/docs/3-design/architecture.md` |
| Load testing strategy | `docs/5-testing/load_testing_strategy.md` |
| Load testing reports | `docs/5-testing/report/load_testing_*.md` |
| Performance reports | `docs/5-testing/report/perf-*.md` |
| Operations guide | `docs/7-operations/operations_guide.md` |
| Deployment guide | `docs/6-deployment/deployment_guide.md` |
| Developer guide | `docs/4-development/developer_guide.md` |
| Inference pipeline | `docs/3-design/guides/inference_pipeline.md` |
| Glossary | `docs/3-design/glossary.md` — semaphore, RAII, KV cache, quantization, etc. |

## License

MIT OR Apache-2.0
