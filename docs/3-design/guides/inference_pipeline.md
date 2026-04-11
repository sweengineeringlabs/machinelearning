# Inference Pipeline — How Text Generation Works

This guide traces a single prediction through the inference stack, mapping each step to the crate that implements it.

## Input → Output

Given input `"The cat sat"`, the model predicts the next token `" on"`.

## Pipeline

```
"The cat sat"
     │
     ▼
┌─────────────────────────────────────────┐
│ EMBEDDING (embedding/)                  │
│                                         │
│  Token IDs: [The]=0  [cat]=1  [sat]=2   │
│  Look up 3 rows from weight matrix      │
│  → 3 dense vectors, each 768 floats     │
└─────────────────────────────────────────┘
     │  shape: [3, 768]
     ▼
┌─────────────────────────────────────────┐
│ NORMALIZATION (normalization/)          │
│                                         │
│  RMSNorm: divide each vector by its RMS │
│  Stabilizes values before attention     │
│  [0.23, -1.5, 0.8, ...] → [-0.3, ...]  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ ATTENTION (inference/layers/)           │
│                                         │
│  Q = input × W_q  (what am I seeking?)  │
│  K = input × W_k  (what do I contain?)  │
│  V = input × W_v  (what is my value?)   │
│                                         │
│  ┌─── ROPE (inference/layers/rope.rs) ─┐│
│  │ Rotate Q and K by position angle    ││
│  │ so the model knows token order:     ││
│  │  pos 0 → rotate 0°                 ││
│  │  pos 1 → rotate θ°                 ││
│  │  pos 2 → rotate 2θ°                ││
│  │                                     ││
│  │ SIMD: AVX2 processes 8 floats/cycle ││
│  │ Same math, 8x throughput            ││
│  └─────────────────────────────────────┘│
│                                         │
│  Scores = Q × K^T / sqrt(d)            │
│                                         │
│  "sat" attends to:                      │
│    "The" → 0.1  (low relevance)         │
│    "cat" → 0.7  (high — what sat?)      │
│    "sat" → 0.2  (moderate — self)       │
│                                         │
│  Output = softmax(scores) × V           │
│  → weighted mix of all token values     │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ NORMALIZATION (normalization/)          │
│  RMSNorm again — stabilize before FFN   │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ FEED-FORWARD / MLP (inference/layers/) │
│                                         │
│  ACTIVATION (activation/):              │
│    gate = SiLU(input × W_gate)          │
│    SiLU adds nonlinearity:              │
│    silu(x) = x × sigmoid(x)            │
│                                         │
│  up  = input × W_up                     │
│  out = (gate * up) × W_down             │
│                                         │
│  QUANTIZED KERNEL (quant/):             │
│    Weights stored as 4-bit integers     │
│    SIMD dot product: Q4 × Q8            │
│    4x less memory bandwidth             │
└─────────────────────────────────────────┘
     │
     ▼
  (...repeat for each layer...)
     │
     ▼
┌─────────────────────────────────────────┐
│ OUTPUT (inference/model/)               │
│                                         │
│  hidden × embedding^T → logits          │
│  logits[" on"] = 8.3  ← highest         │
│  logits["the"] = 2.1                    │
│  → predict " on"                        │
└─────────────────────────────────────────┘
```

## Layer Count

Each model repeats the normalization → attention → normalization → FFN block a fixed number of times:

| Model | Layers | Parameters |
|-------|--------|------------|
| GPT-2 small | 12 | 124M |
| Gemma 3 1B | 26 | 1B |
| Llama 2 7B | 32 | 7B |
| Mixtral 8x7B | 32 | 47B |

The layer count comes from `ModelConfig.n_layers`, read from the model's config at load time.

## Crate Mapping

| Step | Crate | What it does |
|------|-------|-------------|
| Token → vector | `embedding/` | Row lookup from weight matrix |
| Normalize | `normalization/` | RMSNorm / LayerNorm math |
| Attention | `inference/layers/` | Q/K/V projection, scores, weighted sum |
| Position encoding | `inference/layers/` | RoPE with SIMD (AVX2/NEON) |
| Activation | `activation/` | SiLU, GELU pointwise nonlinearity |
| Quantized matmul | `inference/quant/` | SIMD dot products on 4/8-bit weights |
| KV cache | `inference/layers/` | Pre-allocated key/value buffers for decoding |
| Model assembly | `inference/model/` | Composes layers into LlmModel |
| Text generation | `inference/generation/` | Token-by-token loop, sampling, streaming |
| HTTP API | `inference/daemon/` | Serves /v1/completions and /v1/embeddings |
| Weight loading | `hub/` | Downloads from HuggingFace, loads SafeTensors |
| GGUF loading | `inference/gguf/` | Parses GGUF format model files |
| Tokenization | `inference/tokenizer/` | Text ↔ token IDs (BPE, SentencePiece, HF) |
