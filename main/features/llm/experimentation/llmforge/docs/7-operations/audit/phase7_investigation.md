# Phase 7: Performance & Correctness Investigation

> **Date**: 2026-02-13
> **Baseline**: AMD Ryzen 5 7520U, 8 threads, 6.6GB RAM, WSL2
> GPT-2 124M F32 = 12.0 tok/s | TinyLlama 1.1B Q4_0 = 1.8 tok/s

---

## Investigation Results

| # | Issue | Finding | Action |
|---|-------|---------|--------|
| 1 | Parameter count mismatch (131M vs 1.1B) | **BUG** — `parameter_count()` skips `self.layers` entirely | Fix code |
| 2 | KV cache not wired in | **Already working** — `Generator::generate_stream()` uses `make_cache()` + `forward_with_cache()` | Update backlog only |
| 3 | Q4_0 SIMD dispatch | **Already working** — runtime `is_x86_feature_detected!("avx2")` dispatch in `dot_q4_block()` | Add verification log |
| 4 | Rayon parallelism | **Already working** — `quantized_matmul_q4()` uses `par_chunks_mut()` over output rows | Add verification log |

---

## 1. Parameter Count Mismatch — BUG

**File**: `src/models/mod.rs:434-465`

`LlmModel::parameter_count()` counts only:
- `token_embedding.weight` (32000 x 2048 = 65.5M)
- `output.weight` (32000 x 2048 = 65.5M)
- `norm.weight` + `norm.bias` (~4K)
- **Total = 131.1M** — matches observed output

**Missing**: No iteration over `self.layers` (22 transformer blocks), which contain:
- Per-layer attention: `q_proj`, `k_proj`, `v_proj`, `out_proj` (4 Linear layers)
- Per-layer FFN: `up_proj`, `down_proj`, `gate_proj` (3 Linear layers, SwiGLU)
- Per-layer norms: `attention_norm`, `ffn_norm` (2 LayerNorm)
- **Missing ~969M parameters**

**Model IS loaded correctly** — `from_pretrained()` loops `n_layers=22` times and builds all TransformerBlocks. The bug is purely in the counting method.

---

## 2. KV Cache — Already Working

**Evidence chain**:
1. `Generator::generate_stream()` calls `self.make_cache()` → creates `KVCache` with entries for all layers
2. Prefill: `self.prefill(&tokens, &mut cache)` → calls `model.forward_with_cache()` → processes all prompt tokens, stores K/V
3. Decode loop: `self.decode_step(next_token, &mut cache)` → single-token `forward_with_cache()` → reads full K/V history from cache
4. `MultiHeadAttention::forward_with_cache()` does: project Q/K/V → apply RoPE with `start_pos` → `cache.update()` → `cache.get_view()` → attention over full history

**KV caching IS active during CLI inference** — no changes needed.

---

## 3. SIMD Dispatch — Already Working

**File**: `src/quantization/simd.rs:293-315`

`dot_q4_block()` uses compile-time `#[cfg(target_arch)]` + runtime `is_x86_feature_detected!("avx2")`:

| Priority | Architecture | Path | Throughput |
|----------|-------------|------|-----------|
| 1 | x86_64 | AVX2 (`dot_q4_block_avx2`) | 8 f32/cycle |
| 2 | x86_64 | SSE2 (`dot_q4_block_sse2`) | 4 f32/cycle |
| 3 | aarch64 | NEON (`dot_q4_block_neon`) | 4 f32/cycle |
| 4 | any | Scalar fallback | 1 element/iter |

AMD Ryzen 5 7520U (Zen 2 Mendocino) supports AVX2 → primary path is selected.

**No logging exists** to verify at runtime — will add diagnostic output.

---

## 4. Rayon Parallelism — Already Working

**File**: `src/quantization/mod.rs:336-422`

`quantized_matmul_q4()` uses `output.par_chunks_mut(out_features)` — Rayon parallelization over output rows.

**File**: `src/config.rs:207-224`

`RuntimeConfig::apply()` sets:
- `faer::set_global_parallelism(Parallelism::Rayon(n))` — for F32 matmul via faer
- `rayon::ThreadPoolBuilder::new().num_threads(n).build_global()` — for Q4/Q8 matmul (only when n > 0; when n=0, Rayon auto-detects all cores)

**No logging exists** for thread count — will add diagnostic output.

---

## Remediation Plan

### Fix 1: `parameter_count()` bug

Add `parameter_count() -> (usize, usize)` to:
1. `Linear` — weight + bias
2. `LayerNorm` — weight + bias
3. `MultiHeadAttention` — q/k/v/out_proj
4. `CrossAttention` — q/k/v/out_proj
5. `FeedForward` — up/down/gate_proj
6. `TransformerBlock` — attention + ff + norms
7. `LlmModel::parameter_count()` — add `self.layers` loop

### Fix 2: SIMD + thread verification logging

In `RuntimeConfig::apply()`, log detected SIMD capabilities and active Rayon thread count to stderr.

### Fix 3: Update BACKLOG.md

Close KV cache item (already working). Mark other items with findings.
