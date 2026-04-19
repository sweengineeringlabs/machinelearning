# llmserv

LLM-serving Cargo workspace. Contains the inference compute library plus the
serving frontends (daemon, cli, embedding server) and experimentation code.

This is a nested workspace, separate from the root `machinelearning/` workspace.
See the repo root `README.md` for the overall layout and the
cross-workspace dependency rule.

## Layout

```
llmserv/
├── Cargo.toml                  workspace manifest
├── BACKLOG.md                  remaining work on the inference stack
└── main/
    ├── config/
    │   └── application.toml    canonical bundled config — single source of truth
    └── features/
        ├── inference/          library stack, daemon-agnostic
        │   ├── architectures/  bert, falcon, gemma3, gemma4, gpt2, llama,
        │   │                   mixtral, nomic_bert
        │   ├── compute/        compute-backend trait + CpuBackend
        │   ├── generation/     token sampling, temperature, top-k, top-p
        │   ├── gguf/           GGUF parser and writer
        │   ├── layers/         attention, RoPE, RMSNorm, feed-forward, MoE
        │   ├── model/          LlmModel, ModelConfig, ModelBuilderRegistry
        │   ├── prefill/        prefill-strategy trait
        │   ├── llmkernel/      quantization types + SIMD kernels
        │   ├── quantizer/      runtime quantization strategies
        │   ├── thread-config/  rayon pool configuration
        │   └── tokenizer/      BPE, GGUF, HuggingFace backends
        ├── daemon/             swellmd HTTP chat-completions server
        ├── cli/                sweai multi-command developer CLI
        ├── embedding/server/   swe-ml-embed embedding HTTP server
        └── experimentation/
            └── llmforge/       archived inference prototype (read-only)
```

## Build

```bash
cargo build --release --manifest-path Cargo.toml
```

Binaries land at `llmserv/target/release/`.

## Configuration

Daemons (`swellmd`, `swe-ml-embed`) are entirely config-driven — no CLI
flags. Edit `main/config/application.toml` to change defaults, or provide
an overlay at `$XDG_CONFIG_HOME/llmserv/application.toml`.

Load order (deep-merge, later wins):

1. Bundled default (compiled into the binary via `include_str!`)
2. `$XDG_CONFIG_DIRS/llmserv/application.toml` (each entry)
3. `$XDG_CONFIG_HOME/llmserv/application.toml`

See the daemon architecture doc (`main/features/daemon/docs/3-design/architecture.md`)
for the full lifecycle.

## Dependencies on other workspaces

llmserv depends on the root workspace foundation crates and on the llmmodel
workspace, via relative paths in `Cargo.toml`'s `[workspace.dependencies]`:

- Foundations (root): `swe-ml-tensor`, `swe-ml-normalization`, `swe-ml-activation`,
  `swe-ml-embedding`, `swe-ml-architectures`, `swe-ml-training`
- Model I/O (llmmodel): `swe-llmmodel-download`, `swe-llmmodel-io`, `swe-llmmodel-weights`

The arrow flows one way: **llmserv → llmmodel → root**. Never the reverse within libraries.
