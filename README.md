# MACHINELEARNING

Pure-Rust ML platform with three workspaces: LLM inference, ML SDK, and time series.

## Workspaces

### tensor-engine/ — Shared Tensor Engine

| Crate | Description |
|-------|-------------|
| `tensor-engine` | Multi-dtype tensors, SIMD ops, memory-mapped storage, shapes, runtime config |

Shared foundation consumed by both `llm/` and `ml-sdk/`.

### llm/ — LLM Inference Stack

| Crate | Description |
|-------|-------------|
| `llm/nn` | Neural network layers — attention (MHA/GQA), RoPE, RMSNorm, MoE |
| `llm/quant` | Quantization kernels — Q4_0/Q4_1/Q8_0 with AVX2/NEON SIMD |
| `llm/nlp` | Language models — unified arch for GPT-2, Llama, Gemma 4 |
| `llm/tokenizer` | Tokenizers — BPE, GGUF, HuggingFace backends |
| `llm/gguf` | GGUF format — parser and writer for quantized model weights |
| `llm/hub` | HuggingFace Hub — model download, caching, SafeTensors loading |
| `llm/quantize` | Quantization pipeline — SafeTensors to GGUF conversion |
| `llm/cli` | CLI tools — unified `sweai` binary |
| `llm/daemon` | Inference daemon — `swellmd` HTTP service |

### ml-sdk/ — ML SDK

| Crate | Description |
|-------|-------------|
| `ml-sdk` | Foundational building blocks — autograd tensors, layers, optimizers, losses, ops |

Layers: Linear, Conv1d, LSTM, BatchNorm, LayerNorm, Dropout, Sequential, activations (ReLU, GELU, SiLU, Sigmoid, Tanh).
Optimizers: Adam, AdamW, SGD with gradient clipping and LR schedulers.
Losses: MSE, MAE, CrossEntropy, Huber, Quantile.
Ops: matmul, add, mul, relu, sigmoid, softmax, tanh — all with autograd backward passes.

### timeseries/ — Time Series Training

| Crate | Description |
|-------|-------------|
| `timeseries` | Time series models (N-BEATS, TCN, LSTM, Transformer), training, data pipeline |

Consumes `ml-sdk` for layers, optimizers, and losses. Adds domain-specific models, OHLCV data loading, feature engineering, and the Trainer loop.

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
