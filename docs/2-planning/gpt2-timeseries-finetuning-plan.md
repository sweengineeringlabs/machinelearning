# ADR-004: GPT-2 Time Series Fine-Tuning Bridge

**Status:** Proposed (pending review)
**Date:** 2026-04-04

## Context

GPT-2's transformer architecture can be repurposed for time series forecasting. The core insight is that token sequences and time step sequences are structurally identical — ordered data where past values predict future values. Research (FPT, Zhou et al. 2023; TimeGPT; Lag-Llama) has demonstrated that pre-trained language model weights transfer effectively to time series tasks, even when the transformer layers are kept frozen.

RustML has the components needed for this but they exist in separate systems:

| Crate | Provides | Missing |
|-------|----------|---------|
| **rustml-nlp** | GPT-2 forward pass, pre-trained weights | Gradient computation, training loop |
| **rustml-swets** | Autodiff, optimizers (Adam/SGD), trainer, losses | Transformer model, pre-trained weights |
| **rustml-core** | Tensor ops (matmul, softmax, SIMD kernels) | Gradient tracking |

### The Fundamental Gap

`rustml-swets` and `rustml-nlp` use different tensor paradigms:

```
rustml-core::Tensor          rustml-swets::Tensor
├── data: Vec<u8>            ├── id: TensorId
├── shape: SmallVec          ├── inner: rustml_core::Tensor  ← wraps core
├── dtype: DType             ├── requires_grad: bool
└── (no gradients)           └── (gradients via thread-local tape)
```

Key observations:
- `swets::Tensor` already wraps `rustml_core::Tensor` internally (the `inner` field)
- GPT-2's forward pass (`GptModel::forward`) accepts and returns `rustml_core::Tensor`
- swets' autodiff uses a thread-local tape with explicit `record_op()` calls
- rustml-nn layers (Linear, Attention, etc.) never interact with swets' tape

### Goal

Enable fine-tuning GPT-2 for time series forecasting by bridging the two systems. The model architecture:

```
[B, S, ts_features]  →  input_projection  →  GPT-2 layers  →  output_projection  →  [B, S, forecast_dim]
```

Where `ts_features` is the number of time series input features (e.g. price, volume, volatility) and `forecast_dim` is the prediction output size.

---

## Option A: Adapter Pattern (Recommended)

### Approach

Create a thin `TimeSeriesFineTuner` adapter that:
1. Uses `swets::Linear` for trainable input/output projections (autodiff-aware)
2. Delegates to `GptModel::forward` for frozen transformer layers (no gradients)
3. Converts between tensor types at the projection boundaries

```
swets::Tensor [B, S, ts_features]
    ↓
swets::Linear input_proj (trainable, records to tape)
    ↓
swets::Tensor [B, S, 768]
    ↓  ← .inner.clone() conversion
rustml_core::Tensor [B, S, 768]
    ↓
GptModel.forward() (frozen, no tape recording)
    ↓
rustml_core::Tensor [B, S, 768]
    ↓  ← swets::Tensor::new(core_tensor, requires_grad=true)
swets::Tensor [B, S, 768]
    ↓
swets::Linear output_proj (trainable, records to tape)
    ↓
swets::Tensor [B, S, forecast_dim]
```

### Implementation Plan

#### Phase 1: Adapter Crate Scaffold (Day 1)

Create a new module in `rustml-swets` (or a standalone crate) with the adapter:

**File: `rustml/swets/main/src/core/models/gpt2_ts.rs`**

```rust
pub struct Gpt2TimeSeriesConfig {
    pub ts_input_dim: usize,      // Number of input features (e.g. 7)
    pub forecast_dim: usize,      // Number of output features (e.g. 1)
    pub gpt2_dim: usize,          // GPT-2 embedding dim (768)
    pub seq_len: usize,           // Context window length
    pub freeze_gpt2: bool,        // Always true for Option A
}

pub struct Gpt2TimeSeries {
    config: Gpt2TimeSeriesConfig,
    input_proj: swets::Linear,    // [ts_input_dim, 768] — trainable
    gpt2: GptModel,               // Frozen pre-trained GPT-2
    output_proj: swets::Linear,   // [768, forecast_dim] — trainable
}

impl Layer for Gpt2TimeSeries {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        // 1. Project input features to GPT-2 dimension
        let x = self.input_proj.forward(input)?;

        // 2. Convert to core tensor for frozen GPT-2
        let x_core = x.inner.clone();

        // 3. Forward through frozen GPT-2 (no tape recording)
        let y_core = self.gpt2.forward(&x_core)
            .map_err(|e| SwetsError::ModelError(e.to_string()))?;

        // 4. Convert back to swets tensor
        let y = Tensor::from_core(y_core, /*requires_grad=*/ true);

        // 5. Project to forecast dimension
        self.output_proj.forward(&y)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        // Only return trainable projection parameters
        // GPT-2 weights excluded — they stay frozen
        let mut params = self.input_proj.parameters_mut();
        params.extend(self.output_proj.parameters_mut());
        params
    }
}
```

**Trainable parameters:**
- `input_proj.weight`: [768, ts_input_dim] = 768 × 7 = 5,376 params
- `input_proj.bias`: [768] = 768 params
- `output_proj.weight`: [forecast_dim, 768] = 1 × 768 = 768 params
- `output_proj.bias`: [forecast_dim] = 1 param
- **Total trainable: ~6,913 params** (vs 124M frozen in GPT-2)

#### Phase 2: Tensor Conversion Utilities (Day 1-2)

Add helper methods to `swets::Tensor`:

```rust
impl Tensor {
    /// Wrap a rustml_core::Tensor, optionally enabling gradient tracking.
    pub fn from_core(inner: rustml_core::Tensor, requires_grad: bool) -> Self { ... }

    /// Extract the underlying rustml_core::Tensor (clone).
    pub fn to_core(&self) -> rustml_core::Tensor { self.inner.clone() }
}
```

#### Phase 3: Training Integration (Day 2-3)

Wire into existing swets trainer:

```rust
// Load pre-trained GPT-2
let gpt2 = load_gpt2_from_safetensors("openai-community/gpt2")?;

// Build fine-tuner
let config = Gpt2TimeSeriesConfig {
    ts_input_dim: 7,        // e.g. OHLCV + 2 indicators
    forecast_dim: 1,         // predict next close price
    gpt2_dim: 768,
    seq_len: 128,
    freeze_gpt2: true,
};
let model = Gpt2TimeSeries::new(config, gpt2);

// Train
let optimizer = Adam::new(0.001);
let loss_fn = MseLoss::new();
let trainer = Trainer::builder()
    .model(model)
    .optimizer(optimizer)
    .loss(loss_fn)
    .epochs(50)
    .build();

trainer.fit(train_loader, val_loader)?;
```

#### Phase 4: Data Pipeline (Day 3-4)

Create a time series dataset adapter:

```rust
pub struct TimeSeriesDataset {
    windows: Vec<(Tensor, Tensor)>,  // (input_window, target_window)
}

impl TimeSeriesDataset {
    /// Create sliding windows from raw time series data.
    /// input: [total_steps, features]
    /// Returns: Vec<([seq_len, features], [forecast_horizon, forecast_dim])>
    pub fn from_matrix(
        data: &[Vec<f32>],
        seq_len: usize,
        forecast_horizon: usize,
        target_col: usize,
    ) -> Self { ... }
}
```

#### Phase 5: Validation (Day 4-5)

- [ ] Gradient check: verify `tape::grad()` returns non-zero for projection weights
- [ ] Freeze check: verify GPT-2 weights are identical before and after training
- [ ] Smoke test: overfit on a single batch of synthetic sine wave data
- [ ] Benchmark: train on ETTh1 (standard TS benchmark) and report MSE

### Effort Estimate

| Task | Time | Risk |
|------|------|------|
| Adapter struct + Layer impl | 1 day | Low |
| Tensor conversion helpers | 0.5 day | Low |
| Training loop integration | 1 day | Low |
| Data pipeline | 1 day | Low |
| Gradient validation + tests | 1-2 days | Low |
| **Total** | **~5 days** | **Low** |

### Trade-offs

**Pros:**
- No changes to GptModel, rustml-core, or rustml-nn
- Tape recording works for projections automatically
- GPT-2 freezing is automatic (weights excluded from `parameters_mut()`)
- Existing swets trainer, optimizer, loss functions work unchanged
- Minimal new code (~200-300 lines)

**Cons:**
- Tensor copy at projection boundaries (~2 copies per forward pass)
  - For [B=32, S=128, D=768]: ~12 MB per copy, 4 copies per step = ~48 MB
  - Negligible vs GPT-2's 500 MB weight memory
- Only input/output projections are trainable — cannot fine-tune attention layers
- GptModel uses O(n²) attention (no KV cache) — acceptable for training with teacher forcing but slower than necessary

---

## Option B: Full Autodiff Integration

### Approach

Make `rustml-core` operations autodiff-aware so gradients flow through GPT-2's transformer layers. This enables fine-tuning any layer, not just projections.

### Implementation Plan

#### Phase 1: Gradient-Aware Core Operations (Weeks 1-2)

Add `BackwardOp` implementations for every `rustml_core::Tensor` operation used in GPT-2's forward pass:

```rust
// For each op, implement the backward function:
struct MatmulBackward { saved_a: Tensor, saved_b: Tensor }
impl BackwardOp for MatmulBackward {
    fn backward(&self, grad_output: &Tensor, grads: &mut GradMap) -> Result<()> {
        // grad_a = grad_output @ b.T
        // grad_b = a.T @ grad_output
    }
}

struct SoftmaxBackward { saved_output: Tensor }
struct GeluBackward { saved_input: Tensor }
struct LayerNormBackward { saved_input: Tensor, saved_mean: Tensor, saved_var: Tensor }
struct AddBackward;
struct ReshapeBackward { original_shape: Vec<usize> }
// ... ~15 more ops
```

Operations requiring backward implementations:
| Op | Gradient Complexity | Notes |
|----|-------------------|-------|
| matmul | Medium | Two matrix multiplies |
| add | Trivial | Pass-through |
| softmax | Medium | Jacobian-vector product |
| gelu | Easy | Element-wise derivative |
| layer_norm | Hard | 3 intermediate tensors saved |
| reshape/transpose | Trivial | Shape-only, no data op |
| mul_scalar | Trivial | Scalar multiply |
| residual_add | Trivial | Fan-out gradient |
| embedding_lookup | Medium | Sparse gradient scatter |
| causal_mask | Easy | Zero-masked gradient |

**Estimated: ~20 BackwardOp implementations, ~2000 lines**

#### Phase 2: Tape-Aware Wrappers for rustml-nn (Weeks 2-3)

Modify `rustml-nn` layers to optionally record to swets' tape:

```rust
// Either:
// (a) Add feature flag to rustml-nn that enables tape recording
// (b) Create wrapper layers in a new crate (rustml-train)
// (c) Rewrite swets layers to use rustml-nn internals

// Example: tape-aware Linear
impl TapeAwareLinear {
    fn forward(&self, input: &swets::Tensor) -> swets::Tensor {
        let output = input.matmul(&self.weight.t())?;  // records MatmulBackward
        let output = output.add(&self.bias)?;            // records AddBackward
        output
    }
}
```

#### Phase 3: GptModel Training Variant (Weeks 3-4)

Create a training-mode GptModel where every operation records to the tape:

```rust
pub struct TrainableGptModel {
    // Same structure as GptModel but with swets::Tensor weights
    wte: swets::Embedding,
    wpe: swets::Embedding,
    blocks: Vec<TrainableGptBlock>,
    ln_f: swets::LayerNorm,
}

impl Layer for TrainableGptModel {
    fn forward(&mut self, input: &swets::Tensor) -> SwetsResult<swets::Tensor> {
        // Full forward pass, all ops recorded to tape
        let h = self.wte.forward(input)?;
        let h = h.add(&self.wpe.forward(&pos_ids)?)?;
        for block in &mut self.blocks {
            h = block.forward(&h)?;
        }
        self.ln_f.forward(&h)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        // ALL parameters — or selectively freeze layers
        // by excluding them here
    }
}
```

#### Phase 4: Selective Layer Freezing (Week 4)

```rust
pub struct FreezeConfig {
    pub freeze_embeddings: bool,
    pub freeze_layers: Vec<usize>,  // e.g. [0..10] freeze first 10, train last 2
    pub freeze_ln_f: bool,
}
```

#### Phase 5: Testing & Validation (Weeks 5-6)

- [ ] Gradient check for every BackwardOp (numerical vs analytical)
- [ ] Full backward pass through 12-layer GPT-2
- [ ] Memory profiling (tape + saved tensors can be large)
- [ ] Correctness: compare gradients with PyTorch reference
- [ ] Training convergence on ETTh1 benchmark

### Effort Estimate

| Task | Time | Risk |
|------|------|------|
| BackwardOp implementations (~20 ops) | 2 weeks | Medium — correctness critical |
| Tape-aware rustml-nn wrappers | 1-2 weeks | Medium — tight coupling risk |
| TrainableGptModel | 1 week | Medium |
| Selective freezing | 0.5 week | Low |
| Gradient checks + validation | 1-2 weeks | High — subtle numerical bugs |
| Integration testing | 1 week | Medium |
| **Total** | **~6-8 weeks** | **High** |

### Trade-offs

**Pros:**
- Any layer trainable (full fine-tuning, LoRA-style, layer freezing)
- No tensor copies at boundaries
- Foundation for training other models (Llama, Gemma) in the future
- Proper training framework comparable to PyTorch

**Cons:**
- 6-8 weeks of work vs 1 week for Option A
- High risk of subtle gradient bugs (softmax, layer_norm backward are error-prone)
- Major refactoring across 3 crates (core, nn, swets)
- Tight coupling between inference and training code paths
- Memory overhead: tape stores all intermediate tensors for backward pass
  - GPT-2 12 layers × [B=32, S=128, D=768] activations ≈ 1.2 GB saved tensors
- Risk of breaking the inference-optimized code path (SIMD, quantization)

---

## Comparison

| Criteria | Option A (Adapter) | Option B (Full Integration) |
|----------|-------------------|---------------------------|
| **Effort** | ~1 week | ~6-8 weeks |
| **Risk** | Low | High |
| **Trainable layers** | Input/output projections only | Any layer |
| **Code changes** | New adapter only (~300 lines) | 3 crates modified (~3000+ lines) |
| **Inference impact** | Zero (GptModel untouched) | Risk of regression |
| **Memory overhead** | ~48 MB copies per step | ~1.2 GB saved activations |
| **Research backing** | FPT (2023): frozen + projections is competitive | Standard fine-tuning approach |
| **Future extensibility** | Limited to frozen base | Full flexibility |
| **Time to first result** | Days | Months |

---

## Recommendation

**Start with Option A.** The research evidence (FPT, Zhou et al. 2023) shows that frozen pre-trained transformers with trainable projections achieve competitive results on time series benchmarks. Option A delivers a working system in ~1 week with minimal risk.

**Revisit Option B only if:**
1. Option A's forecasting accuracy is insufficient on target datasets
2. There is a clear need to fine-tune attention layers (e.g. domain-specific temporal patterns not captured by pre-trained weights)
3. The training framework needs to support other model architectures beyond GPT-2

Option B is valuable long-term infrastructure but is not justified until Option A's limitations are empirically demonstrated.

---

## Appendix: GptModel Forward Pass (Reference)

The operations that would need BackwardOp in Option B:

```
input_ids [B, S]
    ↓  embedding_lookup
token_embedding [B, S, 768]
    ↓  embedding_lookup
position_embedding [B, S, 768]
    ↓  add
hidden [B, S, 768]
    ↓
┌─── 12x GptBlock ─────────────────────────────┐
│  ↓  layer_norm (weight, bias)                 │
│  ↓  matmul (Q = h @ Wq, K = h @ Wk, V = h @ Wv) │
│  ↓  reshape [B, heads, S, 64]                │
│  ↓  matmul (QK^T)                            │
│  ↓  mul_scalar (1/sqrt(64))                  │
│  ↓  causal_mask + softmax                    │
│  ↓  matmul (attn @ V)                        │
│  ↓  reshape [B, S, 768]                      │
│  ↓  matmul (out_proj)                        │
│  ↓  residual_add                             │
│  ↓  layer_norm (weight, bias)                │
│  ↓  matmul (c_fc) → gelu → matmul (c_proj)  │
│  ↓  residual_add                             │
└───────────────────────────────────────────────┘
    ↓  layer_norm (ln_f)
    ↓  matmul (wte.T) — tied embedding
logits [B, S, 50257]
```

**Total ops requiring backward: ~18 distinct operations, applied 12x per layer.**

## See Also

- [GPT-2 Architecture Guide](../guides/GPT-2.md) — Model architecture and weight mapping
- [Developer Guide](../../4-development/developer_guide.md) — Crate architecture and SEA layering
- [ADR-001: Unified LlmModel](adr-001-unified-llmmodel-for-gpt2.md) — GptModel vs LlmModel decision
