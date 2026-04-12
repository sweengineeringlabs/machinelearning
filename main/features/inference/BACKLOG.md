# rustml Performance Optimization Backlog

> Generated from profiling session 2026-02-19. Based on GPT-2 and Gemma 3 1B inference with `RUST_LOG=rustml=debug`.

## Optimization Methodology

**Always baseline before optimizing.** Capture current metrics before making changes to measure actual improvement.

### Workflow
1. **Baseline** — Run profiling, save metrics to file
2. **Analyze** — Identify bottleneck from trace data
3. **Implement** — Make targeted optimization
4. **Measure** — Re-run profiling, compare to baseline
5. **Document** — Record before/after in `docs/7-operations/audit/`

### Baseline Commands

**Linux/macOS:**
```bash
# Capture baseline (save to dated file)
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0 \
  2>&1 | tee baseline_$(date +%Y%m%d).log

# Key metrics to extract
grep "model::forward" baseline.log      # Step times
grep "transformer\[" baseline.log       # Layer times
grep "\[attn\]" baseline.log            # Attention breakdown
grep "prefill\|decode_step" baseline.log # Prefill vs decode
```

**Windows PowerShell:**
```powershell
# Capture baseline (save to dated file)
$env:RUST_LOG="rustml=debug"
cargo run --release -p rustml-nlp --bin rustml-infer -- `
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0 `
  2>&1 | Tee-Object -FilePath "baseline_$(Get-Date -Format yyyyMMdd).log"

# Key metrics to extract
Select-String "model::forward" baseline.log      # Step times
Select-String "transformer\[" baseline.log       # Layer times
Select-String "\[attn\]" baseline.log            # Attention breakdown
Select-String "prefill|decode_step" baseline.log # Prefill vs decode
```

> **Tip:** Run 3-5 times, discard first (cold cache), average the rest. System variance is typically 10-20%.

---

## Profiling Summary

### GPT-2 (163M params)
- **Config**: 12 layers, 768 dim, 25 Q8_0 layers (attention F32 - below threshold)
- **Fusions**: None (768 < 1024, biased projections)

| Component | Steady-State | Variance | % of Total |
|-----------|--------------|----------|------------|
| Embedding | ~0.1ms | 0.09-0.16ms | <1% |
| Layers (12) | ~46ms | 40-56ms | **~90%** |
| Norm | ~0.8ms | 0.8-1.6ms | ~2% |
| Projection | ~3ms | 2.0-3.9ms | ~6% |
| **Total** | ~50ms | 44-61ms | 100% |

**Per-layer**: Attention ~0.9ms (50%), FFN ~0.8ms (50%)

### Gemma 3 1B (1.3B params)
- **Config**: 26 layers, 1152 dim, 183 Q8_0 layers (all quantized)
- **Fusions**: 26 gate+up pairs, 26 QKV triples

| Component | Steady-State | Variance | % of Total |
|-----------|--------------|----------|------------|
| Embedding | ~0.3ms | 0.2-0.5ms | <1% |
| Layers (26) | ~175ms | 168-185ms | **~87%** |
| Norm | ~0.02ms | 0.01-0.03ms | <1% |
| Projection | ~22ms | 17-64ms | ~11% |
| **Total** | ~198ms | 185-206ms | 100% |

**Per-layer**: Attention ~1.7ms (15%), FFN ~10ms (85%)

---

## Comparative Analysis

| Metric | GPT-2 | Gemma 3 |
|--------|-------|---------|
| Attention:FFN ratio | 50:50 | 15:85 |
| Step jitter | 1.4x | 1.1x |
| Projection jitter | 2x | **3.7x** |
| QKV quantized | No | Yes |
| QKV fused | No | Yes |
| Gate+up fused | N/A | Yes |

**Key insight**: Fusions work well on Gemma 3 (attention is fast). The bottleneck shifts to FFN for larger models.

---

## Tasks

### 1. Add trace-level profiling for matmul operations
**Priority**: High | **Status**: Done | **Blocks**: #2, #4, #6

Add `RUST_LOG=rustml=trace` output for each matmul call with dimensions, timing, and memory bandwidth.

**Expected output format**:
```
[TRACE] matmul Q_proj [1,768]x[768,768] 0.15ms (12.5 GB/s)
```

**Files**: `llm/quant/main/src/core/quantize.rs`, `llm/nn/main/src/core/linear.rs`

---

### 2. Investigate attention quantization threshold for smaller models
**Priority**: High | **Status**: ✓ Done

Lowered Q8_0 threshold from 1024 → 768 to enable attention quantization for GPT-2.

**Results (2026-02-19)**:
| Metric | F32 (before) | Q8_0 (after) | Change |
|--------|--------------|--------------|--------|
| Layers quantized | 25 | 73 | +48 |
| Generation time | 0.54s | 0.49s | **-10%** |
| Output quality | ✓ | ✓ | Identical |
| Attention memory | 4x | 1x | **-75%** |

**Conclusion**: Q8_0 at 768 dim is beneficial — faster inference and lower memory with no accuracy loss.

**Files**: `llm/nlp/main/src/core/model.rs`

---

### 3. Reduce timing jitter from rayon scheduling
**Priority**: Medium | **Status**: ✓ Improved

GPT-2 layer times vary 1.4x within same run. Gemma 3 shows lower jitter (1.1x).

**Solution implemented (2026-02-19)**:
Added `RuntimeConfig::warmup_thread_pool()` called from `warmup_decode()`:
- Forces all rayon threads to spawn and do work before timed inference
- Warms instruction cache with SIMD code paths
- Touches memory to populate TLB entries

**Results**:
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Layer max time | 13.2ms | 5.4ms | **-59%** |
| Layer variance | 5.1x | 2.5x | **-51%** |
| Step variance | 1.9x | 1.4x | **-26%** |
| Step range | 39-75ms | 35-49ms | Tighter |

**Remaining jitter** (~1.4x) is due to OS scheduling and cache effects that cannot be eliminated in software.

**Files**: `llm/core/main/src/core/runtime.rs`, `llm/nlp/main/src/core/model.rs`

---

### 4. Profile and optimize attention forward pass
**Priority**: Medium | **Status**: ✓ Profiled (near-optimal)

Attention optimization is model-dependent:
- GPT-2: 50% of layer time (F32 projections, no fusion)
- Gemma 3: 15% of layer time (Q8_0 + QKV fusion working well)

**Trace profiling results (2026-02-19)**:

| Component | GPT-2 (768 dim) | Gemma 3 (1152 dim) | % of Attention |
|-----------|-----------------|--------------------| --------------|
| QKV projection | 0.3-0.4ms | 0.5-0.6ms | **50-60%** |
| QK normalization | N/A | 0.04-0.06ms | 5% |
| RoPE | N/A | 0.015ms | <2% |
| QK^T matmul | 0.08ms | 0.08-0.12ms | 10-15% |
| Softmax | 0.006ms | 0.007ms | <2% |
| A*V matmul | 0.012ms | 0.015ms | <2% |
| Output projection | 0.1ms | 0.2-0.3ms | 18-25% |
| **Total** | **0.5-0.6ms** | **0.9-1.0ms** | 100% |

**Findings**:
1. QKV projection dominates (~55% of attention time)
2. Core attention ops (QK^T + softmax + A*V) are fast (~0.1ms total)
3. QKV fusion is working well on Gemma 3
4. Output projection is 2nd largest cost

**Optimization candidates evaluated**:
- ✗ Flash attention (fused QK^T + softmax + AV): Core ops are already <0.1ms combined — not worth complexity
- ✗ SIMD softmax: Already <0.01ms per layer
- ✗ Better KV cache: `get_view()` is <0.002ms

**Conclusion**: Attention is near-optimal. Projections dominate, and these are already:
- Q8_0 quantized (GPT-2 after threshold lowering)
- QKV fused (Gemma 3)
- Output projection could theoretically be fused with residual add, but gains would be minimal (~0.1-0.2ms per layer)

**Files**: `llm/nn/main/src/core/attention.rs`, `llm/nn/main/src/core/kv_cache.rs`

---

### 5. Fix lm_head projection variance
**Priority**: Medium | **Status**: Resolved (trace analysis)

Projection times show significant variance, especially on larger models:
- GPT-2: 2.0-3.9ms (2x variance)
- Gemma 3: 17-64ms (**3.7x variance!**)

**Root cause (confirmed via trace)**: Variance is between prefill (multi-token) and decode (single-token), not cold cache.

**Trace-level findings**:
- Warmup: 20.5ms @ 15.7 GB/s
- Prefill (2 tokens): 41.8ms @ 7.7 GB/s (expected: ~2x for 2 tokens)
- Decode: 17.8ms @ 18.0 GB/s (excellent bandwidth)

**Conclusion**: No optimization needed. The 3.7x variance was comparing 2-token prefill vs 1-token decode. Single-token decode is consistent at ~18ms with excellent bandwidth (18 GB/s).

**Files**: `llm/nlp/main/src/core/model.rs`, `llm/nn/main/src/core/linear.rs`

---

### 6. Optimize FFN for large models
**Priority**: Low | **Status**: Investigated — Near-optimal

FFN dominates layer time on Gemma 3 (85% vs 15% attention). Gate+up fusion helps but FFN is still the bottleneck.

**Trace-level findings (per layer)**:
| Operation | Avg Time | Bandwidth | Utilization |
|-----------|----------|-----------|-------------|
| Gate+Up fused | 1.299ms | 13.9 GB/s | 99% |
| Down proj | 0.794ms | 8.9-11.1 GB/s | 64-79% |
| **Total FFN** | **2.09ms** | — | — |

**Investigation results (2026-02-19)**:
- Tested block tiling to improve L1 cache reuse → **Failed** (2x slower due to vec allocation overhead)
- Root cause: Down_proj input (27KB) doesn't fit in L1 (32KB), requires L2 access
- Gate+up input (4.5KB) fits in L1, hence higher utilization
- This is a fundamental cache size constraint, not code inefficiency

**Remaining optimization paths** (diminishing returns):
1. Hardware prefetching (CPUs already do this automatically)
2. Weight layout restructuring (invasive, would affect all models)
3. GPU acceleration (different architecture)

**Conclusion**: Current implementation is near-optimal for CPU. The 20-35% gap from Gate+Up is due to L1 vs L2 cache access patterns inherent to the different input sizes.

**Files**: `llm/nn/main/src/core/feed_forward.rs`, `llm/quant/main/src/core/quantize.rs`

---

### 7. Prefill optimization for longer contexts
**Priority**: Low | **Status**: ✓ Investigated (already efficient)

**Updated findings (2026-02-19)**:

Prefill is actually **more efficient** per-token than decode:

| Model | Prefill | Decode | Prefill/token |
|-------|---------|--------|---------------|
| GPT-2 (13 tokens) | 500ms | 50ms | **38ms** (24% faster) |
| Gemma 3 (10 tokens) | 1604ms | 178ms | **160ms** (10% faster) |

**Per-layer trace analysis (Gemma 3, 14 tokens prefill vs 1 token decode)**:

| Component | Prefill (14 tok) | Decode (1 tok) | Scaling |
|-----------|------------------|----------------|---------|
| QKV projection | 10.4ms | 0.5ms | ~14x (linear ✓) |
| QK^T matmul | 3.5ms | 0.2ms | ~17x (O(n²) expected) |
| A*V matmul | 0.98ms | 0.02ms | ~49x (O(n) × output) |
| Softmax | 0.06ms | 0.01ms | ~6x |

**Why prefill is efficient**:
1. Better memory bandwidth utilization (larger batches)
2. Better cache locality (sequential token access)
3. Linear ops dominate (QKV projection is 60%+ of attention)
4. O(n²) attention scales acceptably for typical prompt lengths

**Conclusion**: No optimization needed. Prefill already processes tokens more efficiently than decode due to batching benefits. The original concern was based on comparing cold-start prefill vs warm decode.

**Files**: `llm/nlp/main/src/core/generator.rs`, `llm/nn/main/src/core/attention.rs`

---

## Recommended Order

1. ~~**#1** - Foundation for data-driven decisions~~ ✓ Done
2. ~~**#5** - Quick win, huge impact on Gemma 3~~ ✓ Resolved (no action needed)
3. ~~**#6** - FFN optimization~~ ✓ Investigated (near-optimal)
4. ~~**#2** - Enable Q8_0 for GPT-2 attention~~ ✓ Done (10% speedup)
5. ~~**#3** - Reduce timing jitter~~ ✓ Improved (51% layer jitter reduction)
6. ~~**#4** - Attention optimizations~~ ✓ Profiled (near-optimal, core ops <0.1ms)
7. ~~**#7** - Prefill optimization~~ ✓ Investigated (already efficient, faster per-token than decode)

## Expected Impact

| Task | Model | Status | Expected Improvement |
|------|-------|--------|---------------------|
| #1 Trace profiling | All | ✓ Done | Enables data-driven optimization |
| #2 Lower Q8_0 threshold | GPT-2 | ✓ Done | 10% faster, 75% less memory |
| #3 Jitter fix | GPT-2 | ✓ Improved | 51% layer jitter reduction |
| #4 Attention optimization | Both | ✓ Profiled | Near-optimal (core ops <0.1ms combined) |
| #5 lm_head variance | Gemma 3 | ✓ Resolved | N/A (was measurement artifact) |
| #6 FFN optimization | Gemma 3 | Near-optimal | N/A (L1 cache constraint) |
| #7 Prefill optimization | All | ✓ Investigated | Already efficient (10-24% faster per-token than decode) |

---

### 8. Fix Gemma 4 E2B forward pass — garbled output from GGUF
**Priority**: High | **Status**: Open | **Blocks**: Gemma 4 inference

Gemma 4 E2B GGUF inference produces garbled/repetitive output (e.g. "気の気の気の"). Confirmed with both our custom-quantized GGUF and the official `ggml-org/gemma-4-E2B-it-GGUF` Q8_0 — proving the issue is in the forward pass, not quantization.

**Infrastructure complete** (2026-04-07):
- Gemma 4 detection from GGUF (`gemma4` arch or `global_head_dim`/`layer_types` present)
- Weight mapping: official + custom GGUF tensor names → internal names
- Config bridge: extracts head_dim, global_head_dim, layer_types, RoPE params, KV sharing, PLE
- Routes to `from_pretrained_gemma4()` builder

**Suspected root causes** (investigate in order):

1. **Per-layer `layer_output_scale`** — Weight loaded from GGUF (`blk.N.layer_output_scale.weight`) and mapped to `layers.N.layer_scalar`, but **never consumed in forward pass**. Gemma 4 multiplies each layer's output by this scalar before the residual add. Without it, residual magnitudes are wrong.
   - Files: `nn/main/src/core/transformer_block.rs` (forward_with_cache), `nlp/main/src/core/model.rs` (from_pretrained_gemma4)

2. **Per-layer feed-forward dimensions** — Official GGUF has `gemma4.feed_forward_length` as a **per-layer array** (len=35). Our code reads a single scalar `hidden_dim`. Some layers may have different FFN widths.
   - Files: `gguf/main/src/core/parser.rs` (to_model_config), `nlp/main/src/core/model.rs`

3. **RoPE partial rotation on global layers** — HF config says `partial_rotary_factor: 0.25` for global layers. GGUF metadata has `rope.dimension_count=512` and `key_length=512`, making the parser compute `512/512=1.0` (full rotation). May need to derive partial factor from `dimension_count / (2 * key_length)` or hardcode for gemma4 arch.
   - Files: `gguf/main/src/core/parser.rs` (rope_parameters extraction)

**Debug approach**: Compare logits on a single token between safetensors path (HF weights) and GGUF path. The first divergence reveals the bug.

**Test files**:
- Official: `ggml-org/gemma-4-E2B-it-GGUF` Q8_0 (4.97 GB)
- Custom: quantized via `rustml-quantize --model google/gemma-4-e2b-it --target q4_0`

---

### 9. Quantize engine: fix summary label "kept F32" for F16 tensors
**Priority**: Low | **Status**: Open

The quantization summary prints "Skipped (kept F32)" but skipped tensors are actually kept as F16. Update the log message in `quantize/main/src/core/engine.rs`.

---

## Inference Speed Improvements

### P0: F16 matmul overhead (RESOLVED)
- **Root cause**: F32 matmul already uses SIMD via faer. F16 was slow because `linear.rs` converted F16→F32 on every forward call. The real fix: quantize F16→Q8_0 at load time (now default via quantization.toml), and allow Q8/Q4 quantizers to accept F16 input directly.
- **Status**: Fixed. Quantizers now auto-convert F16/BF16→F32 before quantizing.

Added 2026-04-11. Gemma 3 1B runs ~1-2s/token on CPU. These optimizations target that bottleneck.

### P1: Aggressive quantization for SafeTensors path
- **Architecture done**: `rustml-quantizer` crate with `Quantizer` trait, `ConfigQuantizer` provider, wired into daemon/cli
- **Remaining**: Write a `quantization.toml` that targets Q4_0 for attention/FFN weights. Test quality impact vs Q8_0. Investigate why SafeTensors loads as F16 instead of applying Q4_0 SIMD kernels.
- **Expected impact**: ~2x speedup (halved memory bandwidth)

### P2: Verify fused QKV + fused gate+up activation
- **Architecture done**: `QkvFuser` and `GateUpFuser` implement `Fuser` trait, wired into daemon GGUF and SafeTensors paths
- **Remaining**: Profile to verify fusion triggers on all architectures (Gemma 3, Llama, Falcon). Log fusion counts per model. Ensure fusion runs after quantization.
- **Expected impact**: 2-3x fewer matmul dispatches per layer

### P3: Batch prefill optimization
- **Architecture done**: `rustml-prefill` crate with `Prefill` trait, `BatchPrefill` provider
- **Remaining**: Profile `forward_pass()` vs `forward_with_cache_pass()` during prompt processing. Verify batch matmul is actually used (not token-by-token). Wire `BatchPrefill` into generator.
- **Expected impact**: Linear speedup for prompt processing (not decode)

### P4: Rayon thread pool tuning
- **Architecture done**: `rustml-thread-config` crate with `ThreadConfig` trait, `AutoThreadConfig` provider, logged at daemon startup
- **Remaining**: Benchmark different thread counts on target hardware. Tune `matmul_parallel_threshold` and `softmax_parallel_threshold`. Add CLI flag `--threads N`.
- **Expected impact**: Variable, depends on core count and workload

### P5: Memory-mapped weights (native in gguf/ and hub/) — DONE
- `GGUFFile::load_tensors_mmap()` in `gguf/`, `load_safetensors_mmap()` in `hub/`, `Tensor::from_mmap()` and `Storage::MMap` in `tensor/`. Zero-copy for F32/F16/Q8/Q4 tensors.
- Embedding server, daemon, and CLI all use mmap. Old `SafeTensorLoader` (read-all-to-F32) deleted.
- **Remaining**: Measure startup time and RSS difference.

### P6: GPU acceleration (Vulkan)
- **Architecture done**: `rustml-compute` crate with `ComputeBackend` trait, `CpuBackend` provider (wraps existing ops), `VulkanBackend` stub
- **Remaining**: Implement Vulkan compute shaders for matmul, softmax, GELU, SiLU. Buffer management, shader compilation pipeline, device selection. Wire `ComputeBackend` into `inference/layers/` to replace CPU tensor ops.
- **Expected impact**: 10-50x speedup for matmul-bound workloads
- **Complexity**: Major project

---

## Test Commands

```bash
# GPT-2 profiling (debug level - layer/step timings)
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0

# GPT-2 profiling (trace level - attention component breakdown)
RUST_LOG=rustml=trace cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 5 --temperature 0 2>&1 | grep "\[attn\]"

# Gemma 3 1B profiling (requires HF_TOKEN)
HF_TOKEN=xxx RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 5 --temperature 0
```
