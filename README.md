# RUSTML

Pure-Rust ML inference framework with quantized LLM serving.

## Workspace

| Crate | Description |
|-------|-------------|
| `rustml/core` | Tensor engine — multi-dtype, SIMD ops, memory-mapped storage |
| `rustml/nn` | Neural network layers — attention (MHA/GQA), RoPE, RMSNorm, MoE |
| `rustml/quant` | Quantization kernels — Q4_0/Q4_1/Q8_0 with AVX2/NEON SIMD |
| `rustml/nlp` | Language models — unified arch for GPT-2, Llama, Gemma 4 |
| `rustml/tokenizer` | Tokenizers — BPE, GGUF, HuggingFace backends |
| `rustml/gguf` | GGUF format — parser and writer for quantized model weights |
| `rustml/hub` | HuggingFace Hub — model download, caching, SafeTensors loading |
| `rustml/cli` | CLI tools — unified `sweai` binary |
| `rustml/daemon` | Inference daemon — `swellmd` HTTP service |
| `rustml/swets` | Time series training — N-BEATS, TCN, LSTM, Transformer |

## Features

- Quantized inference: Q4_0, Q4_1, Q8_0, F16 with SIMD-accelerated kernels
- Model architectures: GPT-2, Llama, Gemma 4 (PLE, logit capping)
- SafeTensors and GGUF model loading with HuggingFace Hub integration
- TOML-configurable per-layer quantization strategy
- Text generation with temperature, top-k, nucleus sampling
- KV caching for efficient autoregressive decoding

## Quick Start

```bash
# Build all crates
cargo build --workspace

# Run inference
cargo run -p rustml-nlp --bin rustml-infer -- --model <path-to-gguf>

# Quantize a model (SafeTensors -> GGUF)
cargo run -p rustml-quantize -- --model <hf-repo> --target q8_0 --output model.gguf

# Inspect a GGUF file
cargo run -p rustml-gguf --bin gguf_inspect -- <path-to-gguf>
```

## Documentation

See `docs/` for architecture decisions, implementation guides, and performance audits.

## License

MIT OR Apache-2.0
