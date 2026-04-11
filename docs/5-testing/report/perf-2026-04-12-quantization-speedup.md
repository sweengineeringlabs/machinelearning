# Performance Report: Quantization + Fusion Speedup

> **Date**: 2026-04-12
> **Platform**: Windows 11, x86_64, AVX2
> **Model**: google/gemma-3-1b-it (1.3B params, SafeTensors)
> **Embedding model**: nomic-embed-text-v1.5 (137M params, GGUF Q8_0)

## Summary

3x inference speedup on Gemma 3 1B by switching quantization config from F16 to Q8_0 and enabling weight fusions. Mmap loading added for GGUF models.

## Root Cause

The `quantization.toml` config was set to `f16` for all layer types. F16 weights in `inference/layers/linear.rs` triggered a per-forward-call conversion path:

```rust
// BEFORE: every forward() call on F16 weights did this
let weight_f32 = self.weight.to_f32()?;  // allocate + convert entire weight matrix
let weight_t = weight_f32.t()?;
x.matmul(&weight_t)?
```

This created a new F32 tensor on every call — for every linear layer, every token. Meanwhile, the Q8_0 path uses pre-quantized SIMD dot products with zero per-call allocation.

The F32 matmul already uses SIMD via the faer library. The issue was never missing SIMD kernels — it was the F16→F32 conversion overhead.

## Changes

### 1. Quantization config: F16 → Q8_0

**File**: `quantization.toml`

```toml
# BEFORE
[quantization]
attention = "f16"
feed_forward = "f16"
output = "f16"
gate = "f16"
moe = "f16"

# AFTER
[quantization]
attention = "q8_0"
feed_forward = "q8_0"
output = "q8_0"
gate = "q8_0"
moe = "q8_0"
```

**Effect**: SafeTensors weights go F32 → Q8_0 at load time. The Q8_0 matmul path uses AVX2 SIMD dot products — no per-call allocation.

### 2. F16 quantizer fix

**File**: `inference/layers/main/src/core/linear.rs`

```rust
// BEFORE: F16 weights silently skipped by quantizer
pub fn quantize_weight_q8(&mut self) -> NnResult<()> {
    if self.weight.dtype() != DType::F32 {
        return Ok(());  // F16 skipped — stuck in slow path
    }

// AFTER: F16/BF16 auto-converted before quantization
pub fn quantize_weight_q8(&mut self) -> NnResult<()> {
    if self.weight.dtype() == DType::F16 || self.weight.dtype() == DType::BF16 {
        self.weight = self.weight.to_f32()?;  // one-time conversion at load
    }
    if self.weight.dtype() != DType::F32 {
        return Ok(());
    }
```

Same fix applied to `quantize_weight_q4_0` and `quantize_weight_q4_1`.

### 3. Weight fusion (already existed, now triggered)

With Q8_0 weights, the fusion methods activate:

- **QKV fusion**: 3 separate Q/K/V projections → 1 fused matmul per layer
- **Gate+Up fusion**: 2 separate gate/up projections → 1 fused matmul per layer

```
Gemma 3 1B:
  Quantized 183 linear layers
  Fused gate+up projection (2 matmuls → 1): 26 pairs
  Fused QKV projection (3 matmuls → 1): 26 triples
```

### 4. Mmap loading for GGUF

**File**: `inference/gguf/main/src/core/parser.rs`

Added `GGUFFile::load_tensors_mmap()` — memory-maps the GGUF file and creates tensors backed by `Arc<Mmap>`. Zero data copy for F32/F16/Q8_0/Q4_0/Q4_1 tensors.

Embedding server switched to mmap loading. Inference daemon still uses eager loading (to be switched).

## Results

### Gemma 3 1B (SafeTensors, chat completions)

| Metric | Before (F16) | After (Q8_0 + fusion) |
|--------|-------------|----------------------|
| Quantization | 183 layers → F16 | 183 layers → Q8_0 |
| QKV fusions | 0 | 26 |
| Gate+up fusions | 0 | 26 |
| Matmuls per layer | 7 | 3 |
| Speed | ~1 tok/s | ~3 tok/s |
| **Speedup** | — | **3x** |

### nomic-embed-text-v1.5 (GGUF Q8_0, embeddings)

| Metric | Before (eager load) | After (mmap) |
|--------|-------------------|--------------|
| Loading | Read entire file into Vec<u8>, copy per tensor | Mmap file, zero-copy tensor references |
| Startup | ~3s | ~3s (model construction dominates at this size) |
| Embedding quality | Identical | Identical (same weights, same computation) |

## Verification

```bash
# Embedding server (mmap)
RUST_LOG=info target/release/swe-ml-embed nomic-embed-text-v1.5.Q8_0.gguf --port 8092
curl -s http://127.0.0.1:8092/v1/embeddings -d '{"input":"test","model":"nomic"}'
# ✓ Returns 768-dim embedding vector

# Inference daemon (Q8_0 + fusion)
RUST_LOG=info target/release/swellmd --safetensors google/gemma-3-1b-it --port 8093
curl -s http://127.0.0.1:8093/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":15}'
# ✓ Returns coherent response, ~3 tok/s

# Streaming
curl -sN http://127.0.0.1:8093/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":20,"stream":true}'
# ✓ SSE events with per-token deltas
```

## Remaining Opportunities

| Item | Expected Impact | Status |
|------|----------------|--------|
| Q4_0 quantization | ~2x over Q8_0 (half bandwidth) | Config change + test quality |
| Mmap for inference daemon | Faster startup for large models | One-line switch |
| Rayon thread tuning | Variable | Needs profiling |
| Vulkan GPU | 10-50x | Major project |
