# GPT-2: Architecture and Role in the LLM Pipeline

> Understanding GPT-2 as a foundation model — what it is, what it can and cannot do, and how it relates to instruction-tuned models.

## What GPT-2 Is

GPT-2 (Generative Pre-trained Transformer 2) is a **base language model** released by OpenAI in 2019. It was trained on a single objective: **next-token prediction**. Given a sequence of tokens, predict the most likely next token.

Training data: ~40 GB of web text (WebText dataset — outbound links from Reddit posts with 3+ karma).

| Variant | Parameters | Layers | Heads | Embedding Dim | Context |
|---------|-----------|--------|-------|---------------|---------|
| Small | 124M | 12 | 12 | 768 | 1024 |
| Medium | 355M | 24 | 16 | 1024 | 1024 |
| Large | 774M | 36 | 20 | 1280 | 1024 |
| XL | 1.5B | 48 | 25 | 1600 | 1024 |

RustML uses GPT-2 Small (124M) as the default test model.

## Architecture

GPT-2 is a decoder-only transformer with:

```
Input tokens
    ↓
Token Embedding + Learned Positional Embedding
    ↓
┌─────────────────────────────────┐
│  Transformer Block (x N layers) │
│  ├─ LayerNorm                   │
│  ├─ Multi-Head Self-Attention   │
│  ├─ Residual Connection         │
│  ├─ LayerNorm                   │
│  ├─ FFN (up_proj → GELU → down) │
│  └─ Residual Connection         │
└─────────────────────────────────┘
    ↓
LayerNorm
    ↓
Linear (lm_head) → logits [vocab_size=50257]
    ↓
Softmax → next token probabilities
```

**Key architectural choices:**
- **Learned positional embeddings** (not RoPE — that came later with Llama)
- **LayerNorm with bias** (later models like Llama use RMSNorm without bias)
- **Standard FFN** with GELU activation (not gated variants like SwiGLU/GeGLU)
- **No grouped-query attention** — all heads have their own K/V projections
- **Tied embeddings** — token embedding and lm_head share weights

## What GPT-2 Can Do

GPT-2 is a **text completion engine**. It excels at continuing text in a statistically plausible way:

| Task | Input | Output | Quality |
|------|-------|--------|---------|
| Story continuation | "It was a dark and stormy night" | Continues the narrative | Good |
| Text completion | "The capital of France is" | "Paris, a city known for..." | Good |
| Code completion | `def fibonacci(n):` | Plausible continuation | Moderate |
| Few-shot pattern matching | "cat → gato, dog → perro, house →" | "casa" | Moderate |
| Feature extraction | Any text | Internal layer representations | Good |

## What GPT-2 Cannot Do

As a base model, GPT-2 has no concept of instructions, conversations, or user intent:

| Task | Why It Fails |
|------|-------------|
| Follow instructions | Never trained on "instruction → response" pairs |
| Chat / conversation | Doesn't understand user/assistant roles |
| Answer questions | Completes text, doesn't reason about content |
| Refuse harmful content | No safety alignment training |
| Use chat templates | Treats template tokens as regular text |

**Example failure:**

```
Input:  "Hello, how are you?"
Expect: "I'm doing well, thanks for asking!"
Actual: "____type: ma-doo.com\n\ngui: ____ ____type: ma-doo.com"
```

GPT-2 doesn't interpret "Hello" as a greeting requiring a response. It treats it as the start of a document and continues with whatever patterns it learned from web text.

## Base Models vs Instruction-Tuned Models

Every modern chatbot starts as a base model and goes through additional training stages:

```
Stage 1: Pre-training (base model)
    │  Train on raw text with next-token prediction
    │  Result: GPT-2, Gemma 3 base, Llama base
    │  Capability: text completion, pattern matching
    ↓
Stage 2: Supervised Fine-Tuning (SFT)
    │  Train on curated (instruction, response) pairs
    │  Result: understands "do X" → produces X
    │  Capability: follows instructions
    ↓
Stage 3: Alignment (RLHF / DPO)
    │  Train with human preference feedback
    │  Result: prefers helpful, harmless, honest responses
    │  Capability: coherent conversation, safety
    ↓
Instruction-tuned model
    Gemma 3 IT, Llama Chat, ChatGPT, Claude
```

| Property | Base Model (GPT-2) | Instruction-Tuned (Gemma 3 IT) |
|----------|-------------------|-------------------------------|
| Training objective | Next-token prediction | Next-token + instruction following + alignment |
| Input interpretation | Continuation of a document | User request to respond to |
| Chat templates | Ignored (treated as text) | Understood and followed |
| Output quality | Statistically plausible text | Coherent, relevant responses |
| EOS behavior | Rarely stops on its own | Stops when answer is complete |
| Safety | None | Trained to refuse harmful requests |

## Why RustML Uses GPT-2

GPT-2 serves as the **primary test and development model** for several reasons:

1. **Freely available** — no gated access, no license restrictions, no HF token required
2. **Small and fast** — 124M params loads in seconds, generates at ~15 tok/s on CPU
3. **Well-understood** — extensively documented architecture, easy to verify correctness
4. **Architecture blueprint** — GPT-2's transformer block is the foundation for all modern LLMs
5. **Pipeline validation** — if GPT-2 produces tokens, the same code path works for Gemma, Llama, and Mixtral

GPT-2 validates the inference engine. Instruction-tuned models validate the output quality.

## GPT-2 in the RustML Codebase

### Model Implementations

- **`LlmModel`** (`rustml-nlp`) — production unified model, handles GPT-2 via `ModelConfig` dispatch. Uses KV cache for O(n) per-token decode.
- **`GptModel`** (`rustml-nlp`) — reference/teaching implementation. O(n^2) per token, no KV cache. Exists for learning and verification, not production use.

### Loading GPT-2

```rust
// SafeTensors (HuggingFace)
let model = build_safetensors_model("gpt2", &config, weights)?;

// The "gpt2" model_type triggers:
// - Learned positional embedding (not RoPE)
// - LayerNorm with bias (not RMSNorm)
// - Standard FFN (not SwiGLU/GeGLU)
// - Tied lm_head weights
```

### Weight Mapping

GPT-2 SafeTensors weights use HuggingFace naming conventions that are remapped to RustML's internal format:

| HuggingFace Name | RustML Name | Notes |
|-----------------|-------------|-------|
| `wte.weight` | `token_embedding` | Token embeddings |
| `wpe.weight` | `position_embedding` | Positional embeddings |
| `h.{N}.attn.c_attn.weight` | `blk.{N}.attn_qkv.weight` | Fused QKV (Conv1D, transposed) |
| `h.{N}.mlp.c_fc.weight` | `blk.{N}.ffn_up.weight` | FFN up projection (Conv1D) |
| `h.{N}.mlp.c_proj.weight` | `blk.{N}.ffn_down.weight` | FFN down projection (Conv1D) |
| `h.{N}.ln_1.weight` | `blk.{N}.attn_norm.weight` | Pre-attention LayerNorm |
| `h.{N}.ln_2.weight` | `blk.{N}.ffn_norm.weight` | Pre-FFN LayerNorm |
| `ln_f.weight` | `output_norm.weight` | Final LayerNorm |

**Important**: GPT-2 uses Conv1D (not Linear), so weight matrices are transposed relative to standard format. The weight mapper handles this automatically.

### Quantization

At load time, GPT-2's F32 weights are quantized to Q8_0:
- 73 of 160 tensors quantized (linear layers with dim >= 768)
- ~75% memory reduction
- ~10% inference speedup from reduced memory bandwidth

### Performance on CPU

Benchmarked on Windows 11, AVX2, 8 threads (release build, Q8_0):

| Metric | Value |
|--------|-------|
| Typical throughput | ~15 tok/s |
| Per-token latency | ~65ms |
| Model load time | ~15s (including quantization) |
| Memory footprint | ~200 MB |

See [Performance Testing](../../5-testing/perf_testing.md) for full benchmark methodology and results.

## How GPT-2 Differs from Modern Architectures

| Feature | GPT-2 (2019) | Llama/Gemma (2023+) |
|---------|-------------|-------------------|
| Position encoding | Learned embedding | RoPE (rotary) |
| Normalization | LayerNorm (pre-norm) | RMSNorm |
| Bias terms | Yes (all layers) | No |
| FFN activation | GELU | SwiGLU / GeGLU |
| Attention | Full MHA | Grouped-Query (GQA) |
| Context length | 1024 | 8192+ |
| Embedding tying | Yes (wte = lm_head) | No |
| Vocab size | 50,257 | 256,000+ |

These differences are handled transparently by `LlmModel`'s config-driven dispatch — the same forward pass code supports all architectures.

## See Also

- [FFN Architectures](ffn_architectures.md) — Standard FFN vs SwiGLU vs GeGLU
- [Model Verification Guide](../../4-development/guides/model-verification.md) — Verifying GPT-2 correctness
- [ADR-001: Unified LlmModel](../adr/adr-001-unified-llmmodel-for-gpt2.md) — Why GptModel was unified into LlmModel
- [Performance Testing](../../5-testing/perf_testing.md) — Benchmark results
- [Tokenizer & Weight Integration](tokenizer_weight_integration.md) — Weight loading pipeline
