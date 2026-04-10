# MACHINELEARNING

Pure-Rust ML platform with two workspaces: LLM inference and ML SDK.

## Workspaces

### llm/ — LLM Inference Stack

| Crate | Description |
|-------|-------------|
| `llm/core` | Tensor engine — multi-dtype, SIMD ops, memory-mapped storage |
| `llm/nn` | Neural network layers — attention (MHA/GQA), RoPE, RMSNorm, MoE |
| `llm/quant` | Quantization kernels — Q4_0/Q4_1/Q8_0 with AVX2/NEON SIMD |
| `llm/nlp` | Language models — unified arch for GPT-2, Llama, Gemma 4 |
| `llm/tokenizer` | Tokenizers — BPE, GGUF, HuggingFace backends |
| `llm/gguf` | GGUF format — parser and writer for quantized model weights |
| `llm/hub` | HuggingFace Hub — model download, caching, SafeTensors loading |
| `llm/quantize` | Quantization pipeline — SafeTensors to GGUF conversion |
| `llm/cli` | CLI tools — unified `sweai` binary |
| `llm/daemon` | Inference daemon — `swellmd` HTTP service |

### timeseries/ — Time Series

| Crate | Description |
|-------|-------------|
| `timeseries` | Training framework — N-BEATS, TCN, LSTM, Transformer, optimizers |

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
