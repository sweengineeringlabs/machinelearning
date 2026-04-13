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
- `GGUFFile::load_tensors_mmap()` in `gguf/`, `load_safetensors()` in `hub/`, `Tensor::from_mmap()` and `Storage::MMap` in `tensor/`. Zero-copy for F32/F16/Q8/Q4 tensors.
- Embedding server, daemon, and CLI all use mmap. Old `SafeTensorLoader` (read-all-to-F32) deleted.
- **Remaining**: Measure startup time and RSS difference.

### P7: Close the Ollama performance gap (gemma-3-1b, CPU, chat completion)

**Baseline measurement (2026-04-12, N=15 per cell, warmup matched):**

| Server | Quantization | p50 | p95 | wall (15 calls) |
|---|---|---|---|---|
| swellmd | Q8_0 (runtime) | 2699 ms | 3191 ms | 41.4 s |
| swellmd | Q4_0 (runtime) | 2744 ms | 3392 ms | 42.3 s |
| ollama  | Q4_K_M         |  761 ms |  810 ms | 11.4 s |

**Ollama is 3.5× faster on the same workload and hardware.**

**Measurement insight — what we thought vs what the data says:**

Theory predicted ~2× of the gap came from quantization (Q4 uses half the
memory bandwidth of Q8) and ~2× from llama.cpp kernel maturity. We
tested this by running swellmd with Q4_0 instead of Q8_0. Result:
**identical speed** (2744 ms vs 2699 ms). Switching formats gave us
zero speedup.

That means our Q4 matmul doesn't capture the bandwidth savings that
smaller weights should give. Most likely cause: our matmul dequantizes
to F32 before multiplying, so the output-side F32 bandwidth dominates
and compressing the input doesn't matter. Llama.cpp avoids this with
fused `Q × F32 → f32` dot products in AVX2 that never materialize
a dequantized weight tensor.

**Implication:** K-quant format support ALONE will not close the gap.
The bottleneck is kernel quality, not quantization format.

**Prioritized work:**

**P7.1 — Profile the hot path.** `cargo flamegraph` on a single
chat completion run. Confirm matmul dominates (it should). Identify
exactly which variant (Q8 × F32, Q4 × F32, F32 × F32) is the hot
function and how much time is inside unpack vs dot-product vs
accumulate. Without this data we're guessing.

**P7.2 — Fused Q8_0 × F32 dot product.** Hand-written AVX2 loop:
load 32 int8 weights, multiply by scale, dot with 32 F32 activations,
horizontal-sum into accumulator. No intermediate F32 weight tensor.
This is the single biggest expected speedup — our current Q8 matmul
is the most-used kernel (quantization default), so any win here scales
across the whole model.

**P7.3 — Fused Q4_0 × F32 dot product.** Same as P7.2 but with 4-bit
unpacking. Requires P7.2 as a template.

**P7.4 — Fused attention (flash-attention-style).** Compute
`softmax(QK^T / √d) V` without materializing the full attention
matrix in memory. Relevant for longer contexts; less important for
8-token benchmarks. Do after P7.2 / P7.3.

**P7.5 — K-quant format support (Q4_K_M).** ONLY after P7.3. With
P7.3 in place, adding K-quant gets us llama.cpp's on-wire format
compatibility and the further per-sub-block quality improvement.
Without P7.3, it's wasted effort.

**P7.6 — KV cache pooling.** Small win. Reuse per-request caches via a
slab allocator instead of allocating fresh per request.

**Expected overall impact (ORIGINAL, now refined by measurement)**: see
"Status update 2026-04-12" below. P7.2.a delivered ~6%. P7.2.b as
originally specified did NOT deliver the hoped 2–3× on its own.

### P7.2 status update (2026-04-12) — measured, not speculative

**P7.2.a shipped (commit 9309043):** FMA + 4 independent accumulators
in `dot_q8_block_avx2_fma`. Measured: **p50 2699 → 2540 ms, ~6% speedup.**
Correctness verified by existing test_q8_dot_scalar_vs_dispatch.

**P7.2.b infrastructure shipped (commits 463f6e5, 1640358), default off:**
- New `matmul_f32_q8_v2` using integer-domain dot product
- New `dot_q8q8_block_avx2` (signed i8×i8 → i32 via madd_epi16)
- New `quantize_q8_0_into` (allocation-free variant)
- New `use_native_q8` flag on Linear (routes v1 vs v2)
- 2 new tests (test_q8_matmul_v2_matches_v1, test_q8q8_dot_extremes)

**Enabled measurement:** use_native_q8 = true → p50 **2728 ms**. That's
**7% SLOWER** than P7.2.a. The "integer-domain beats FMA" hypothesis did
not hold on this CPU / with this implementation.

**Why it didn't win:** three compounding reasons, none trivial to fix:

1. **Throughput parity, not advantage.** `_mm256_madd_epi16` is single-
   port (1 op/cycle) on Skylake+/Zen+. `_mm256_fmadd_ps` is dual-port
   (2 ops/cycle). v1 already extracts FMA's full ILP via 4 independent
   accumulator chains (from P7.2.a); v2 has a single madd chain.
2. **Allocation hypothesis was wrong.** Per-call `Vec<u8>` in
   `quantize_q8_0` is ~13 μs/token total, not the explanation.
3. **Extra per-block scalar work.** Two scale f16→f32 + scalar multiply
   per block, vs v1's single scale multiply.

**What would actually deliver a speedup from v2** (unshipped, estimated):

- **P7.2.b.1 — Multi-accumulator integer kernel.** Mirror the 4-
  accumulator pattern from P7.2.a inside dot_q8q8. Likely requires
  restructuring to process 4 output columns in parallel per activation
  block (since scales differ per block, we can't keep integer
  accumulators across blocks).
- **P7.2.b.2 — Cross-block SIMD f32 accumulator.** Use a single __m256
  f32 accumulator living across blocks for one output column; reduce
  once per column, not per block. Saves 5 hsum ops per block × 36 blocks
  × ~6912 columns per FFN matmul ≈ 1.2M SIMD ops saved per matmul.

Realistic effort estimate: ~1 week of focused SIMD work + benchmarking
for each of .b.1 and .b.2. Expected combined speedup: plausibly 1.5–2×
on the matmul kernel, closing ~30–50% of the remaining Ollama gap. Not
certain — measurement trumps estimate.

**Alternative concrete path to close the gap (recommended for triage):**
wrap llama.cpp's `ggml_mul_mat` as a `ComputeBackend` provider (the trait
already exists from P6). Effort: ~1-2 weeks including FFI binding, build
integration, CI. Outcome: parity with llama.cpp on the kernels that matter,
at the cost of a C++ toolchain dependency. Ships concrete performance
parity faster than reimplementing their SIMD in Rust.

**Recommendation going forward:** accept current performance (p50 2540 ms,
3.3× slower than ollama) unless raw speed is a hard requirement. If it
is, pursue the llama.cpp-wrapper path (P7.x tbd) rather than continuing
to hand-tune Rust SIMD — the same engineer-time delivers larger and
more predictable gains wrapping a mature kernel library.

### P7.2 decision framing — concrete work per path

If picking up this thread in a future session, the four realistic paths
with their actual work items:

**Path A — Continue hand-tuned SIMD (P7.2.b.1 + P7.2.b.2).**

  Work (5–8 days focused):
  - Restructure outer matmul to process 4 output columns per activation
    block; write dot_q8q8_4cols_avx2 with 4 independent madd chains
  - Handle out_features % 4 != 0 edge cases
  - New correctness tests (4× code paths, same tolerance)
  - Cross-block __m256 f32 accumulator in inner loop (reduce once per
    output column, not per block)
  - Benchmark each change independently, keep only wins

  Expected outcome:
  - Optimistic: 20–40% additional speedup, gap shrinks to 2.3–2.6×
  - Realistic: 10–20%, gap shrinks to 2.7–3.0×
  - Pessimistic: no net win (measurement reveals the optimizations
    don't pay on this CPU/workload)

  Further P7 items (P7.3 Q4_0, P7.4 fused attention, P7.5 Q4_K_M,
  P7.6 KV cache pooling) each add 1–2 weeks more. Ceiling to reach
  within ~1.5× of ollama: 2–3 months.

**Path B — Wrap llama.cpp as a ComputeBackend (recommended if speed
matters).**

  Work (8–12 days):
  - llama.cpp as git submodule OR use llama-cpp-rs / llama-cpp-2
    crate to skip most FFI (2 days)
  - Implement LlamaCppBackend satisfying the existing ComputeBackend
    trait from P6 (2 days)
  - CI build on Linux/Windows/macOS for the C++ side (1–2 days)
  - Wire [compute] section in application.toml to pick backend (0.5 day)
  - Correctness tests against current backend, same prompts same model (1 day)
  - Benchmark vs current CpuBackend (0.5 day)

  Risk: low. llama-cpp-rs exists, is MIT-licensed, actively maintained.
  Expected outcome: ~3× speedup, effectively matching ollama.

  Trade-offs:
  - Adds C++ toolchain dependency for source builds
  - Loses the "pure Rust" identity
  - Gains kernels for ARM / Apple Silicon / AVX-512 (currently AVX2 only)

**Path C — Accept current performance, move on (status quo).**

  Work: 0 days. Net cost: opportunity cost of continuing with
  2540 ms p50 on a 1B model. Fine for dev / desktop / prototyping.
  Not competitive with ollama for users who benchmark both.

  Use if: other backlog items (D1 doc sweep, T1 oha parity P0/P1,
  or P5 GPU Vulkan via P6) have higher product impact than the
  remaining 3.3× perf gap.

**Path D — Optimize something other than matmul.**

  Not recommended: profiling showed matmul is 89% of decode-token time.
  Optimizing the other 11% (output projection, sampling, KV cache)
  caps the possible speedup at ~12%. Poor ROI vs Path B.

  Exception: P7.4 fused attention becomes relevant if workloads shift
  to longer contexts (the 8-token benchmark doesn't stress attention).

**Framing for the decision:**

| Path | Work | Risk | Expected speedup | Identity change |
|---|---|---|---|---|
| A | 5–8 days (+ more for P7.3–.6) | medium (uncertain) | 10–40% | none |
| B | 8–12 days | low | ~200% (matches ollama) | adds C++ dep |
| C | 0 | none | 0% | none |
| D | 2–3 weeks | medium | ≤12% | none |

**Alternative "pragmatic" path**: wrap llama.cpp itself as a
`ComputeBackend` provider (we already have the trait from P6). Lose
"pure Rust" as a bragging point, gain every ounce of their kernel
tuning. Consider if the time investment above proves untenable.

---

### P7.B: `LlamaCppBackend` — implementation plan

Concrete build-out of Path B. Goal: a drop-in backend that matches
Ollama's speed (~3× current CpuBackend) by delegating the forward pass
to llama.cpp, selected via `application.toml`.

**Architecture decision — op-level vs model-level wrapping.**

The existing `ComputeBackend` trait at
`llmserv/main/features/inference/compute/src/api/traits.rs` exposes
per-op methods (`matmul`, `softmax`, `gelu`, `silu`). Wrapping
llama.cpp at that granularity defeats the point — llama.cpp's speed
comes from **whole-graph ggml fusion**, per-CPU kernel dispatch, and
its own KV cache layout. Calling `ggml_mul_mat` once per Linear layer
from our Rust code would round-trip buffers across the FFI boundary
dozens of times per token and lose most of the win.

Caveat to record: `ComputeBackend` is not actually plumbed into the
forward pass today. `serve.rs:40` constructs `CpuBackend`, logs its
name, and drops it. `AppState` holds `Box<dyn Model>`, never a
compute backend. Forward pass calls `LlmModel::forward_pass` →
`layer.forward` directly against tensor ops. So "the CpuBackend swap
seam" is nominal, not real — even `VulkanBackend` would have to wire
itself in before it could be swapped.

Conclusion: `LlamaCppBackend` owns the **whole forward pass**, not
individual ops. **The DI seam we need already exists** — it's the
daemon-side `Model` trait at
`llmserv/main/features/daemon/main/src/api/model.rs:8`:

```rust
pub trait Model: Send + Sync {
    fn model_id(&self) -> &str;
    fn build_generator(&self, temperature: f32) -> Generator<'_>;
    fn tokenizer(&self) -> &dyn Tokenizer;
    fn embed(&self, input_ids: &Tensor, strategy: PoolingStrategy) -> ModelResult<Tensor>;
}
```

`AppState.model: Box<dyn Model>` is what axum handlers call.
`load_model()` already returns `Box<dyn Model>`. `LlamaCppBackend`
implements this trait directly — no new abstraction needed.

**Correction to the earlier draft of this plan**: a separate
`ModelBackend` trait in the `llmcompute` crate is redundant. The
daemon-side `Model` trait IS the model-level backend abstraction, and
the existing `Rust*Model` adapter is already a `Model` impl wrapping
our native stack. Drop that layer; implement `Model` for llama.cpp
directly. The `ComputeBackend` trait stays for future per-op Vulkan
work on the native path (if we ever plumb it in).

**Swappability scope.** Any impl of `Model` slots in at this DI seam:
`LlamaCppModel` (this plan), a future `VulkanModel` (native Rust
forward pass but GPU-backed tensor ops), a `RemoteHttpModel` (proxy
to another inference server), a `MockModel` for testing. The daemon
doesn't know or care which backs the trait object.

**Dependency choice.**

Recommended: **`llama-cpp-2` crate** (actively maintained, bundled
llama.cpp submodule, covers Linux/Windows/macOS CPU + CUDA + Metal
features behind cargo flags). Alternatives weighed:

| Option | Pro | Con |
|---|---|---|
| `llama-cpp-2` crate | Maintained, bundles cmake build, covers major OSes | Pulls llama.cpp at a pinned version; we track upstream via crate updates |
| `llama-cpp-rs` crate | Older, smaller API | Less maintained — rejected |
| Manual bindgen + submodule | Full control over version pinning | 3-5 days more work up front, we become cmake-build maintainers |

Pick `llama-cpp-2`. If it becomes unmaintained we can migrate to a
submodule without changing the `LlamaCppBackend` trait surface.

**Work breakdown.**

1. **Clean up dead `ComputeBackend` wire (~0.25 day).**
   - Either delete the unused `let compute = llmcompute::CpuBackend;`
     lines in `serve.rs:40-41`, or plumb the handle into something
     real. Prefer delete — no consumer exists, and `LlamaCppBackend`
     is a `Model`, not a `ComputeBackend`. Vulkan work later can
     re-introduce the seam when there's an actual consumer

2. **Add `llama-cpp-2` dependency (~0.5 day).**
   - `[workspace.dependencies] llama-cpp-2 = { version = "...", features = ["cmake"] }`
   - Confirm build succeeds on the dev box (MSVC toolchain on Windows,
     cmake on PATH)
   - Gate behind a cargo feature `backend-llama-cpp` so pure-Rust
     builds don't require a C++ toolchain

3. **Implement `LlamaCppModel: Model` (~2.5 days).**
   - New crate `llmserv/main/features/inference/backend-llama-cpp/`
     following SEA layout (api/core/saf)
   - Wrap `LlamaModel` load, `LlamaContext` creation, `llama_decode`
     per-batch, logits extraction
   - Implement the daemon's `Model` trait (model.rs:8): `model_id`,
     `build_generator`, `tokenizer`, `embed`
   - Adapt llama.cpp's BPE to our `Tokenizer` trait (thin wrapper over
     `llama_tokenize` / `llama_token_to_piece`) so `tokenizer()` returns
     `&dyn Tokenizer` like the native path — ensures bit-identical
     inputs vs Ollama without forcing the daemon to branch on backend
   - Map llama.cpp errors to `ModelError`

4. **Config wiring + DI (~0.5 day).**
   - Add `[model]` section to `main/config/application.toml`:
     ```toml
     [model]
     backend = "native_rust"   # "native_rust" | "llama_cpp"
     ```
   - `load_model()` in `serve.rs:87` branches on this once at startup
     and returns the appropriate `Box<dyn Model>`. Register backends
     via a `ModelBackendRegistry` (SPI/DI — no match-on-string in
     hot paths, only at load time)

5. **Streaming integration (~1 day).**
   - llama.cpp produces logits; our `generation/` crate handles sampling
     — reuse that layer by wrapping `LlamaCppBackend::forward` output
     into our `Logits` type
   - Alternative: use llama.cpp's built-in sampling via
     `llama_sampler_*`. Defer — keeps sampler parity with
     `NativeRustBackend` so daemon SSE behavior is identical

6. **Correctness harness (~1 day).**
   - New test `llmserv/main/features/backend-llama-cpp/tests/parity.rs`
   - Load same GGUF with both backends, generate 32 tokens at
     temperature=0, assert token sequences match for
     gemma-3-1b-it and qwen2.5-0.5b
   - Not expecting exact logit match (different FMA orders), but
     deterministic token output at temp=0 must match

7. **Benchmark (~0.5 day).**
   - Extend `llmc load` harness to accept `--backend llama_cpp`
   - Report p50/p95/p99 decode latency for the same prompts used in
     the Ollama comparison; record in `docs/5-testing/perf/`

8. **CI updates (~0.5–1 day).**
   - Linux runner: `apt install cmake g++` already standard
   - Windows runner: confirm MSVC + cmake on the GitHub-hosted image
   - macOS runner: Xcode CLI tools already present
   - Split CI matrix: default job builds without `backend-llama-cpp`
     feature (fast, no C++); a separate job builds with the feature
     and runs the parity test

**Total: 8–10 days focused work.** Matches the original Path B estimate.

**Open questions — resolved after docs.rs skim of
`llama-cpp-2` v0.1.143.**

- **Logits per decode step: YES.** `LlamaContext` exposes
  `get_logits() -> &[f32]` and `get_logits_ith(i: i32) -> &[f32]`,
  plus higher-level `candidates_ith(i) -> impl Iterator<LlamaTokenData>`
  and `token_data_array_ith(i) -> LlamaTokenDataArray`. Our sampler
  can consume the raw `&[f32]` directly — parity with the
  native-Rust path's sampling stays exact (same math, just different
  input source). No need to use llama.cpp's built-in sampler.

- **Parallel sequences in one context: YES.** Full sequence-ID API:
  `copy_kv_cache_seq(src, dest, p0, p1)`,
  `clear_kv_cache_seq(src, p0, p1)`,
  `kv_cache_seq_add(seq_id, ...)`,
  `kv_cache_seq_pos_max(seq_id)`,
  `llama_kv_cache_seq_keep(seq_id)`. `LlamaBatch` takes seq_id per
  token. This unblocks the "pool shared context across daemon
  admission slots" strategy from audit finding #3 — preferred for
  memory efficiency over `Arc<Mutex<LlamaContext>>` or fresh-per-call.

- **Chat template: YES, two variants.**
  - `LlamaModel::chat_template(name: Option<&str>) -> LlamaChatTemplate`
    — pulls the template from GGUF metadata.
  - `LlamaModel::apply_chat_template(&LlamaChatTemplate,
    &[LlamaChatMessage], add_ass: bool) -> Result<String, _>` —
    applies it with an optional trailing assistant-turn marker.
  - Also `apply_chat_template_oaicompat` for passing OpenAI-shape
    JSON messages directly. Useful if we ever want to skip our own
    message normalization.
  - No need for the fallback "manually format from GGUF metadata"
    path the original plan considered.

- **License / distribution:** llama.cpp is MIT, `llama-cpp-2` is
  also permissive. No conflicts. Note in `NOTICE` if we ship
  binaries.

**API reference summary (pinned to v0.1.143 for task 8 implementer).**

```rust
// Module structure:
//   context, gguf, llama_backend, llama_batch, model, openai,
//   sampling, timing, token, token_type

// Startup — once per process:
let backend = LlamaBackend::init()?;

// Load model — once per daemon lifetime:
let params = LlamaModelParams::default();
let model = LlamaModel::load_from_file(&backend, &gguf_path, &params)?;

// Per-request context (or pooled — audit #3 decision):
let ctx_params = LlamaContextParams::default().with_n_ctx(n_ctx);
let mut ctx = model.new_context(&backend, ctx_params)?;

// Tokenize:
let tokens: Vec<LlamaToken> =
    model.str_to_token(prompt, AddBos::Always)?;

// Decode batch:
let mut batch = LlamaBatch::new(n_ctx, /* n_seq_max */ 1);
// ... fill batch.add(token, pos, &[seq_id], logits_flag) ...
ctx.decode(&mut batch)?;

// Extract logits for the last token (for external sampling):
let logits: &[f32] = ctx.get_logits_ith(batch.last_logits_index());

// Chat templating:
let template = model.chat_template(None)?;
let prompt = model.apply_chat_template(&template, &messages, true)?;

// Detokenize for streaming:
let mut decoder = encoding_rs::UTF_8.new_decoder();
let piece = model.token_to_piece(token, &mut decoder, false, None)?;
```

**Validation / done-criteria.**

- [ ] Parity test passes: same tokens generated for gemma-3-1b-it
      and qwen2.5-0.5b at temp=0, 32-token continuation, 3 prompts
- [ ] Benchmark shows p50 decode within 20% of Ollama on the same
      model, same prompt, same hardware
- [ ] `cargo build --release` without `backend-llama-cpp` feature
      still succeeds (no C++ toolchain required for pure-Rust build)
- [ ] Daemon hot-reload works: switching `backend` in
      `application.toml` and sending SIGHUP swaps backends cleanly
      without restart
- [ ] Correctness: streaming SSE payload is byte-identical for a
      fixed prompt between old and new backends at temp=0 (requires
      matching sampler — handled in step 5)

**What this deliberately does NOT do.**

- Does not delete the native-Rust path. Keeping it means we still own
  the full stack for teaching/research and for platforms where C++ is
  unwelcome.
- Does not port our SIMD kernels into llama.cpp's GGML framework.
  That's a separate, much larger project.
- Does not change the `ComputeBackend` trait. `VulkanBackend` still
  fits there for future per-op GPU work.

**Pre-start audit findings (post-P7.B.1).**

After shipping the trait-surface refactor (P7.B.1, commit `7b3db42`),
these gaps remain that will surface during LlamaCppModel
implementation. None block starting; all need conscious handling.

1. **`Model::embed` still leaks `Tensor`.** Signature
   `fn embed(&self, input_ids: &Tensor, strategy: PoolingStrategy)
   -> ModelResult<Tensor>` was not part of the completion-path
   cleanup. If LlamaCppModel must serve `/v1/embeddings`, the same
   concrete-type leak returns. Two options when you get there:
   - Narrow scope: `LlamaCppModel::embed -> Err("not supported")`.
     Honest, but the daemon's embedding endpoint 500s for
     llama.cpp-backed models.
   - Extend the refactor: add an `EmbeddingResult` struct wrapping
     `Vec<f32> + shape + dtype`; change the trait signature. Same
     pattern as `CompletionParams`. ~1 day.
   - Recommended: punt during initial P7.B landing. Native-Rust path
     still handles `/v1/embeddings`; llama.cpp path serves
     completions only.

2. **Chat template ownership at `complete_turn_stream`.** Trait
   passes `messages: &[(&str, &str)]` — implementor owns templating.
   Native `Generator` applies templates via `encode_conversation`.
   Verify `llama-cpp-2` exposes `apply_chat_template` bindings; if
   missing, LlamaCppModel formats messages itself (look up model's
   template from GGUF metadata and assemble manually). 30-min API
   skim answers this.

3. **Context lifecycle — `&self` on `open_text_completer`.** The
   returned `Box<dyn TextCompleter + '_>` borrows from `&self`.
   `LlamaModel` (weights) is immutable and `Arc`-friendly;
   `LlamaContext` (KV cache + runtime state) is mutable and
   per-sequence. Three strategies for LlamaCppTextCompleter:
   - Fresh `LlamaContext` per `open_text_completer` call — slow
     (KV allocation is the expensive part).
   - `Arc<Mutex<LlamaContext>>` shared across completers — serializes
     requests, defeats daemon concurrency.
   - Context pool keyed per OS thread or per admission-control slot —
     fastest, most code. Preferred for production.
   Not a trait-surface problem; scopes the step-3 work. Budget extra
   time for whichever strategy you pick.

4. **No `[model].backend` wiring yet.** `load_model()` in `serve.rs:87`
   dispatches only on `ModelSource::Safetensors | Gguf`. Adding the
   llama.cpp path means extending that enum (or adding a sibling
   `[model].backend` key that cross-cuts source). ~0.5 day — include
   in step 4 (config wiring).

5. **Pre-existing test-fixture bug blocks `cargo test -p
   rustml-generation`.** `generator.rs:957` — `ModelConfig` literal
   missing `architecture` field. Not caused by P7.B.1 (reproduced
   on the pre-refactor tree via `git stash`). Parity tests for
   LlamaCppModel will likely live alongside rustml-generation or in
   a new `backend-llama-cpp` crate — either way, fix this first
   (~15 min) so the test binary compiles.

**What "ready to start" means after this audit.**

- Completion path (`/v1/completions`, `/v1/chat/completions`,
  streaming SSE, FFI `llmserv_complete*`): trait surface is clean,
  implementable, no concrete types leak.
- Embedding path (`/v1/embeddings`): still leaks `Tensor`. Defer
  llama.cpp support for this endpoint until #1 is resolved.
- Daemon wiring: one enum extension + one config key away from
  selecting backends at startup.
- Testing: one test-fixture fix away from being able to run the
  rustml-generation test suite where parity tests will land.

**P7.B.2 — post-skeleton session outcome.** All four "ready to start"
rows above are resolved and the architectural wiring is shipped:

| Audit row | Status | Commit / task |
|---|---|---|
| Completion path trait surface | ✅ clean | P7.B.1 (trait refactor, `05086aa`) |
| Embedding path `Tensor` leak | ✅ resolved | task 9 (`3e8dabb`); `Model::embed(&[u32], PoolingStrategy) -> Vec<f32>` |
| Daemon wiring (`[model].backend`) | ✅ wired | task 2 (`3e8dabb`); `ModelBackendLoader` SPI registry |
| Test-fixture fix | ✅ fixed | task 1 (`41710b2`); `architecture` field added |
| Audit #2 chat template | ✅ verified | `apply_chat_template` exists in llama-cpp-2 v0.1.143 |
| Audit #3 context lifecycle | ✅ unblocked | parallel-sequence API confirmed — pool strategy preferred |

Backend-contract crate `llmbackend` extracted, new feature-gated
crate `rustml-backend-llama-cpp` scaffolded with SEA layout. Daemon
feature `backend-llama-cpp` forwards to the inner crate's
`llama-cpp` feature. CI matrix covers both paths
(`.github/workflows/ci.yml`).

**What remains for a working llama.cpp backend.** Only real
`llama-cpp-2` API calls inside `LlamaCppModel` — the skeleton
currently returns "integration pending" errors from load. Blocked
on installing MSVC Build Tools on the dev box so cmake + llama.cpp
can be built locally for verification. Once unblocked, the API
reference block above covers every call the implementation needs.
Parity test and benchmark (done-criteria below) follow.

---

### P7.B.1: Trait-surface refactor — prerequisite for LlamaCppBackend

P7.B assumed `LlamaCppModel` could satisfy the daemon's `Model` trait
cleanly. Audit showed the trait surface leaks native-Rust types:

- `Model::build_generator(temperature) -> Generator<'_>` returns a
  concrete struct from `rustml-generation`, not a trait object.
- `Generator<'a>` holds `&'a dyn LanguageModel`, and `LanguageModel`
  (in `rustml-model/src/api/types.rs:573`) exposes:
    - `forward(&self, input_ids: &Tensor) -> ModelResult<Tensor>`
      — leaks our concrete `Tensor` type
    - `forward_with_cache(&self, input_ids: &Tensor, cache: &mut KVCache)`
      — leaks our concrete `KVCache` from `rustml-inference-layers`
    - `build_kv_cache(&self, max_seq_len) -> KVCache` — same leak
- llama.cpp owns its own tensor representation AND its own KV cache
  internally. A `LlamaCppLanguageModel` impl receiving `&mut KVCache`
  would be forced to ignore it — a semantic lie baked into the trait.

Two paths were considered. **Decision: Option 2 (refactor the trait
surface).** Option 1 (accept the lie + per-step tensor copy) was
rejected because the `KVCache` no-op semantics would be a permanent
landmine for anyone maintaining the trait, and a vocab-sized float
copy per token (50k floats × 20 tok/s = small but pointless memcpy
traffic) compounds with any future work.

**Target trait surface.**

```rust
// New module: llmserv/main/features/inference/generation/main/src/api/completer.rs
pub struct CompletionParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub max_tokens: usize,
    pub deadline: Option<Instant>,
}

pub trait TextCompleter: Send {
    fn complete(
        &mut self,
        prompt: &str,
        params: &CompletionParams,
    ) -> CompletionResult<String>;

    fn complete_stream(
        &mut self,
        prompt: &str,
        params: &CompletionParams,
        callback: &mut dyn FnMut(u32) -> bool,
    ) -> CompletionResult<(usize, usize)>;  // (prompt_tokens, completion_tokens)

    fn complete_turn_stream(
        &mut self,
        messages: &[(String, String)],
        params: &CompletionParams,
        callback: &mut dyn FnMut(u32) -> bool,
    ) -> CompletionResult<(usize, usize)>;
}

// Updated daemon-side Model trait:
pub trait Model: Send + Sync {
    fn model_id(&self) -> &str;
    fn open_text_completer(&self) -> Box<dyn TextCompleter + '_>;  // replaces build_generator
    fn tokenizer(&self) -> &dyn Tokenizer;
    fn embed(&self, input_ids: &Tensor, strategy: PoolingStrategy) -> ModelResult<Tensor>;
}
```

**Design rationale for each choice.**

- **`Box<dyn TextCompleter>` return** — the DI seam. Backends own their
  own KV cache, tokenizer bindings, and decode state internally.
  Neither `Tensor` nor `KVCache` appears in the trait surface.
- **`&mut self` on completer methods** — llama.cpp's `LlamaContext`
  is mutable; our decode loop mutates KV cache state; matches reality.
- **`&mut dyn FnMut(u32) -> bool` instead of `<F: FnMut>`** — generic
  type parameters block `dyn`-compatibility. Dynamic dispatch on the
  callback adds one indirect call per token (~ns) against a
  millisecond-scale decode step. Negligible overhead, buys us the
  trait object.
- **`CompletionParams` struct instead of `.with_top_k(k).with_top_p(p)`
  builder chain** — builder methods taking `self` by value don't work
  on `Box<dyn Trait>`. Moving per-request params into a struct is
  also more explicit at call sites.
- **Model-level defaults stay inside the impl** — EOS, BOS, chat
  template, context length, optimization profile are model-scoped
  config read from `DefaultModel` fields at completer construction,
  NOT passed per-request. `CompletionParams` only carries what the
  HTTP caller actually varies.
- **Remove `LanguageModel` trait from public API** — after this
  refactor, `LanguageModel` is an internal implementation detail of
  the native-Rust completer, not part of the backend contract. No
  `Tensor` / `KVCache` leak to backends.

**Naming fixes folded into this refactor.**

Audit surfaced two related smells:

- `Generator` is a generic nominalized verb (like `Handler`, `Manager`).
  Doesn't say what modality — in a multimodal future (`ImageCompleter`,
  `AudioCompleter`) it squats the namespace. Also clashes with
  `std::ops::Generator` / coroutines.
- `TextGenerator` already exists in the same crate
  (`generation.rs:18`) as an older, simpler tensor-in/tensor-out
  variant. Two types, same concept, different names, same crate —
  violates the "descriptive names, no collisions" rule.
- `build_generator` doesn't "build" anything — it copies a config
  struct. `open_text_completer` reads honestly.

Renames performed as part of this work:

| From | To | Location |
|---|---|---|
| `Generator` (struct) | `DefaultTextCompleter` | `core/generator.rs` → becomes the `TextCompleter` impl |
| `TextGenerator` (old, simpler) | delete | `core/generation.rs` — folded into the new completer or removed; nothing uses its API surface that isn't subsumed |
| `Model::build_generator` | `Model::open_text_completer` | `daemon/api/model.rs:8` + `daemon/core/state.rs:25` |
| `GenerationError` / `GenerationResult` | `CompletionError` / `CompletionResult` | follow trait naming |
| crate `rustml-generation` | `llmtext` (optional, defer) | matches `llmkernel` / `llmcompute` sweep — lower priority, can happen in a separate pass |

**Execution order — each step = one commit, `cargo build --release`
verified between steps.**

1. Add `TextCompleter` trait + `CompletionParams` struct to
   `rustml-generation/src/api/completer.rs`. Export from SAF.
   Additive. No behavior change.
2. Implement `TextCompleter` for the existing `Generator` struct as
   an adapter. Behavior preserved via delegation. Tests still green.
3. Add `Model::open_text_completer()` alongside the old
   `build_generator`. Both exist. Call sites unchanged.
4. Implement `open_text_completer()` on `DefaultModel`
   (`daemon/core/state.rs`) returning a boxed `Generator` as
   `Box<dyn TextCompleter + '_>`.
5. Migrate router.rs call sites (4 places: `router.rs:133`, `:220`,
   `:306`, other chat/completion handlers). Replace
   `.build_generator(t).with_top_k(k).with_top_p(p)...` with
   `CompletionParams { temperature: t, top_k, top_p, ... }` construction
   and `completer.complete_turn_stream(..., &params, &mut cb)`.
6. Remove `Model::build_generator` from the trait. Remove the old
   `Generator`-returning impl from `DefaultModel`.
7. Rename `Generator` → `DefaultTextCompleter`. Delete old
   `TextGenerator` (generation.rs). Rename error types.
8. Delete dead `let compute = CpuBackend;` in serve.rs:40.
9. Now P7.B steps 2–8 (add llama-cpp-2 dep, implement
   `LlamaCppTextCompleter: TextCompleter`, wire config, parity tests,
   benchmark, CI) become straightforward — they implement the same
   trait the native path already satisfies.

**Estimated work:** 2–3 days for the refactor (steps 1–8), before
P7.B's 8–10 day llama.cpp work can start. Total stack: 10–13 days.

**Validation after the refactor (before P7.B step 2 begins):**

- [ ] `cargo build --release` clean across the entire `llmserv/` workspace
- [ ] `cargo test --release` all green
- [ ] Live smoke: `swellmd` serves a completion + streamed completion,
      byte-identical output vs pre-refactor for a fixed seed
- [ ] No `Tensor` or `KVCache` types appear in `daemon/src/api/` or in
      the `TextCompleter` trait surface (mechanical check: grep the
      public API files)

---

### P7.C: Continuous batching in `LlamaCppBackend` — close the c>1 gap

**Motivation — measured, not hypothetical.** `docs/5-testing/perf/llama-cpp-vs-native.md`
documents a multi-client benchmark (gemma3:1b, n=40, same GGUF):

| c | llama_cpp p50 | Ollama p50 | llama_cpp req/s | Ollama req/s |
|---|--------------:|-----------:|----------------:|-------------:|
| 1 |           604 |        708 |            1.64 |         1.26 |
| 2 |          1066 |       1059 |            1.80 |         1.71 |
| 4 |          2124 |   **1646** |            1.87 |     **2.34** |

At c=1 our pool beats Ollama by ~15% p50. At c=4 Ollama pulls 25%
ahead on throughput, 22% on p50. Our latencies roughly double as
concurrency doubles — we're not actually parallelizing.

**Root cause.** The pool shipped in `commit 29137e3` holds N
independent `LlamaContext` instances. Each decode owns its own
llama.cpp internal thread pool. On 8 cores, 4 parallel contexts ×
~8 threads/ctx = 32 threads contending for 8 cores plus L1 cache
thrashing. Ollama almost certainly uses llama.cpp's continuous
batching (one context, N sequences interleaved by `seq_id`, one
`llama_decode` call per tick processes all sequences together).
Our "pool of independent contexts" leaves the entire batching
mechanism on the table.

**Scope of the refactor (3–5 days).**

1. **Scheduler component.** New long-running task that owns a single
   `LlamaContext`, pulls from a request queue, packs active
   sequences into one `LlamaBatch` per tick, calls `llama_decode`
   once per tick. Returns per-sequence tokens through channels.
2. **seq_id lifecycle.** Allocate a free id per request;
   `kv_cache_seq_rm` on EOS/max_tokens/cancel; honor the N-seq
   upper bound (`LlamaContextParams::with_n_seq_max`, default small
   — likely needs bumping to daemon throttle capacity).
3. **Admission control rewrite.** Today: semaphore permit per
   request gates concurrent pool acquires. After: bounded queue
   feeding the scheduler with backpressure when the batch is at
   capacity. `[throttle]` config shape may want revisiting.
4. **Prefill + decode scheduling.** Decode adds 1 token per tick per
   request; prefill needs hundreds of tokens in one hit. Either
   interleave prefill into the same tick (complex) or give prefill
   its own tick cadence (simpler, higher latency spikes). This
   scheduling policy is where Ollama/vLLM spend their tuning effort
   — **getting it right takes iteration, not just implementation**.
5. **`TextCompleter` trait boundary.** Current
   `complete_stream(&mut self, prompt, params, callback)` holds a
   `&mut` on the completer for the whole decode. Continuous batching
   has N requests active against one context; the scheduler needs
   all of them simultaneously. Completer becomes a token-channel
   consumer: scheduler pushes, completer pulls + invokes callback.
6. **Cancellation / backpressure.** HTTP client disconnects must
   detach the seq from the batch promptly, else we waste compute
   generating tokens nobody's reading. Tokio drop-handler wiring.

**Risks specific to this refactor.**

- **c=1 regression.** If the scheduler tick overhead is >10 ms on
  an idle server, we lose our current c=1 advantage over Ollama
  (p50 604 vs 708). The existing pool behavior at c=1 is the gold
  standard to beat; continuous batching must match or exceed it, not
  just win at c=4. Measure before committing to the redesign.
- **Harder debugging.** A bug in one request's output could be a
  scheduler issue affecting unrelated sequences. Current pool
  isolates failures per-context.
- **Scheduling policy tail risk.** p99 is sensitive to prefill
  stalls (a big new request can block ongoing decodes for a tick).
  Measurement under a mixed-size-prompt workload is essential, not
  just repeated identical prompts.

**Validation / done-criteria.**

- [ ] c=1 p50 no worse than current (604 ms on gemma3:1b).
- [ ] c=4 throughput at or above Ollama's (2.34 req/s on same
      workload).
- [ ] p95/p99 at c=4 no worse than Ollama's (~2100 ms).
- [ ] Cancellation: killing `curl` mid-stream stops compute within
      one tick (~100 ms).
- [ ] Mixed-prompt workload: interleaving a 500-token prefill into
      ongoing 20-token decodes doesn't spike p99 beyond 2× the
      steady-state p99.

**What this deliberately does NOT do.**

- Does not touch the native-Rust backend. Continuous batching on
  native is a separate (much bigger) project — would need batched
  matmul, shared KV across sequences, etc.
- Does not add prefix caching / LRU. That's a follow-up after
  continuous batching exposes the seq_id lifecycle.
- Does not target multi-GPU or distributed. Single-host CPU only
  for this task.

**Skip if:** swellmd's only use cases are single-user IDE /
desktop / CLI (the current c=1-optimized pool is already ahead of
Ollama there). Come back to this only when multi-client serving is
a real workload.

---

### P6: GPU acceleration (Vulkan)
- **Architecture done**: `llmcompute` crate with `ComputeBackend` trait, `CpuBackend` provider (wraps existing ops), `VulkanBackend` stub
- **Remaining**: Implement Vulkan compute shaders for matmul, softmax, GELU, SiLU. Buffer management, shader compilation pipeline, device selection. Wire `ComputeBackend` into `inference/layers/` to replace CPU tensor ops.
- **Expected impact**: 10-50x speedup for matmul-bound workloads
- **Complexity**: Major project

---

### P8: Long-context + high-concurrency via KV cache compression (TurboQuant)

**Different axis from P7.** P7 attacks **weight matmul speed** (per-token decode
latency). P8 attacks **KV cache memory and bandwidth** (concurrent request
capacity + long-context decode speed). Orthogonal, stack cleanly, different
bottlenecks solved.

**Reference algorithm:** TurboQuant (Zandieh et al., Google Research + NYU,
arXiv 2504.19874, ICLR 2026). Two-stage data-oblivious vector quantization
for online KV cache compression at 2.5–3.5 bits/coordinate.

**Core pipeline (from paper):**
1. Randomly rotate input vector `x`. Rotation pushes each coordinate toward
   a concentrated Beta distribution (→ Gaussian in high dim) with
   near-independence across coordinates.
2. Apply precomputed optimal scalar Lloyd-Max quantizer per coordinate
   (codebooks generated offline via 1D k-means on the Beta distribution).
3. For unbiased inner products (needed for attention scores): apply `b-1`
   bit Q_mse, then a 1-bit QJL residual transform on `x - Q_mse⁻¹(Q_mse(x))`.

**Published results:**
- 3.5 bits/channel → absolute quality neutrality on long-context tasks
  (needle-in-a-haystack, downstream accuracy)
- 2.5 bits/channel → marginal quality degradation
- >5× KV cache memory reduction at neutral quality
- Within ~2.7× of Shannon's information-theoretic lower bound for MSE;
  ~1.45× at 1 bit
- Benchmarks on H100 GPU; CPU applicability plausible but unmeasured in
  the paper

**Problems this would solve for llmserv:**

1. **Concurrent-request OOM unlock.** v0.8.0 load testing recorded process
   death at 50 concurrent requests due to KV cache expansion (~416 MB/request
   × 50 = 20 GB). Throttle mitigated it by rejecting via 503; TurboQuant
   would let the same machine actually serve those requests (~4 GB at 5×
   compression).
2. **Long-context decode speedup.** For contexts > ~4k tokens, KV read
   bandwidth during attention dominates per-token time. 5× compression =
   ~5× speedup on that phase. Invisible in our current 8-token benchmark.
3. **Deployment density.** Same machine serves more concurrent users, or
   hosts bigger models with the same RAM.

**Problems this does NOT solve:**
- The Ollama p50 gap on short prompts (that's matmul-bound, P7 territory).
- Weight storage size (that's P7.5 K-quant territory).

**Scope (estimate, 2–4 weeks focused):**

- `llmkernel-vector` (new or extend existing `llmkernel`):
  - Hadamard-based fast rotation (O(d log d), accelerator-friendly). Paper
    doesn't pin the rotation family but explicitly targets vectorization;
    Hadamard / WHT is the natural choice.
  - Offline codebook generation (continuous 1D Lloyd-Max for Beta-
    distributed coordinates; store codebooks for b ∈ {2,3,4,5})
  - 1-bit QJL transform (random ±1 projection + sign)
  - Q_mse API: `quantize_vector(x: &[f32]) -> (Bytes, Scale)`
  - Q_prod API: wraps Q_mse + QJL residual
  - Dequantization helpers for both
- `rustml-inference-layers::kv_cache`:
  - New `CompressedKvCache` variant that stores per-token K, V as
    TurboQuant blobs instead of raw F16
  - Decompression on attention read (or keep a recent-tokens F16 window
    for speed, compress older entries — a "rolling compression" variant)
  - Integration with existing KVCache trait

- Configuration in `application.toml`:
  ```toml
  [kv_cache]
  backend = "compressed"      # or "default"
  bits_per_coord = 3.5
  unbiased_inner_product = true   # uses Q_prod vs Q_mse
  ```

- Correctness validation:
  - Perplexity on eval set (WikiText-103 or similar): should match
    F16 baseline at 3.5 bits
  - Long-context retrieval quality: needle-in-a-haystack test at 4k/8k
  - Numerical comparison: K × Q / √d scores from compressed vs
    uncompressed paths within tolerance

- Benchmarking:
  - Memory: measure per-request KV RSS at various context lengths
  - Throughput: decode tok/s at 2k / 8k / 32k context
  - Concurrency: re-run the v0.8.0 load test — does it now serve 50
    concurrent without OOM?

**Risks:**

- **Rotation cost on CPU.** For gemma-3-1b (head_dim=256), full random
  rotation = 65k multiplies per K or V vector. Per decode step × 26 layers
  × 2 (K and V) = ~3.4 M multiplies just for rotation. With fast Hadamard
  (O(d log d) = ~2k ops per rotation), drops to ~100 k ops / decode step,
  trivial. With naive dense random rotation, significant. Paper implies
  Hadamard but doesn't guarantee; we'd need to validate quality holds
  with Hadamard specifically.
- **Attention latency regression.** Dequantizing older K, V tokens on
  every attention read adds per-token cost. For short contexts this could
  make decode slower, not faster. Mitigation: "recent-window" hybrid
  (hot recent tokens uncompressed, cold tail compressed).
- **Perplexity regression risk.** Paper claims neutrality at 3.5 bits but
  on their eval sets. Our specific models (Gemma / Llama / etc.) may show
  different tradeoffs. Correctness validation is non-negotiable before
  making it the default.

**Acceptance criteria:**

- Functional: KVCache trait has a compressed-backend option; unit tests
  for quantize/dequantize roundtrip fidelity pass at 3.5 bits.
- Correctness: perplexity on a 1k-sample eval at 3.5 bits within 1%
  of F16 baseline for gemma-3-1b; needle-in-a-haystack accuracy at
  8k context matches F16 within 2%.
- Performance: decode tok/s at 32k context ≥ 2× the F16 baseline on
  the same machine.
- Memory: per-request KV RSS at max context ≤ 20% of the F16 baseline.
- Gating: `[kv_cache].backend` defaults to `"default"` (F16). Switching
  to `"compressed"` is opt-in until acceptance criteria are met for
  all supported architectures.

**Do NOT start this unless:**
- There's a concrete user / workload requirement for long context or
  high concurrency (currently: the load-testing strategy documents 50-
  concurrent capacity as a goal but nothing else drives it).
- OR P7 (matmul speedup) has landed and we're looking for the next
  non-overlapping optimization target.

**Alternative:** adopt an existing KV cache compression method already
shipping in llama.cpp (KV quantization via q8_0/q4_0, also 2–4× compression).
Simpler than implementing TurboQuant from the paper, gives a meaningful
fraction of the memory benefit, no implementation risk. Worth
benchmarking the simpler option first before committing to the novel
algorithm.

---

## Tooling

### T1: Bring `llmc load` toward oha parity — scoped by real need

`llmc load` (commits `8a293ec` and `2d1bd2f`) is focused on llmserv
testing: CO-correct open-loop via `--rate`, our JSON schema, no extra
install. It is **not** a drop-in `oha` replacement. This item tracks
closing the gap where it actually matters, and explicitly deferring the
rest.

For context on tradeoffs see the glossary's "Coordinated omission"
entry and the `--rate` docs in `llmserv/main/features/cli/src/cmd/load.rs`.

#### Priority 0 — add these (real scenarios need them)

**T1.1 — Body / URL templating with variables.**
Our load tests today send the same body every request. Realistic
benchmarking needs varying prompts (different lengths, different
content) so the server doesn't get a caching-friendly fixed input.
Minimum: support `{{VAR}}` substitution in the URL and body from a
values file or a generator function. Oha supports this.

**T1.2 — Progress output during long runs.**
Currently the tool is silent until the summary at the end. A soak
test at `-z 1h` gives no feedback. Minimum: periodic stderr status
line every N seconds with running count, achieved RPS, p95. No TUI.

**T1.3 — `--insecure` flag for TLS.**
When someone stands up the daemon with a self-signed cert behind a
local reverse proxy, `llmc load https://...` fails with cert error.
Thread through reqwest's `danger_accept_invalid_certs`.

#### Priority 1 — add if a scenario demands it

**T1.4 — Histogram snapshot file output.**
Write the hdrhistogram to a `.hgrm` file (hdrhistogram's native
log format) so runs can be merged across hosts or compared over time
with existing hdrhistogram tooling.

**T1.5 — DNS / connect / TTFB breakdown.**
Separate timings for DNS resolution, TCP connect, TLS handshake,
time-to-first-byte, and body transfer. Requires using hyper directly
(reqwest hides these). Useful when diagnosing why a request is slow.

**T1.6 — HTTP/2 and HTTP/1.1 force-protocol.**
`--http2` / `--http1` to pin the protocol, for reproducing issues
under one or the other. Reqwest supports this behind a builder flag.

#### Priority 2 — deferred or explicit non-goals

- **Full TUI (ratatui dashboard).** Low ROI for our workflow — CI wants
  JSON, dev wants one-shot summaries. Skip unless demand emerges.
- **HTTP/3 / QUIC.** Our daemon doesn't speak it. Add when it does.
- **Cookie / redirect / session control.** We test APIs, not browser
  flows.
- **Distributed multi-host load generation.** Listed as non-goal in
  `docs/5-testing/load_testing_strategy.md` — one client host is
  sufficient at 1B-model CPU inference speeds.
- **Request recording / replay from a pcap or HAR.** Interesting but
  way out of scope.

#### Acceptance

- T1.1, T1.2, T1.3 shipped as incremental commits, each with smoke
  tests against a live daemon.
- Glossary entry updated to reflect added capabilities.
- Docs (`docs/5-testing/load_testing_strategy.md`) mention `llmc load`
  as the canonical driver and note which P1/P2 gaps remain.
- Each P2 deferral has an explicit rationale in this backlog so it
  doesn't get rediscovered and reconsidered silently.

### T2: Ship an FFI / `cdylib` for desktop and IDE integration

Motivated by **deployment ergonomics**, not performance. An IDE plugin
that calls llmserv via HTTP has to manage port binding, process
lifecycle, crash-restart, and firewall prompts on first run. An FFI
path collapses all of that: the plugin process *is* the inference
process, one binary ships, no server to manage.

Be honest about what this does NOT buy:
- Chat completion is ~3000 ms of compute; the ~1 ms HTTP loopback
  overhead is 0.03% of that. FFI does not make generation faster.
- Embedding single texts shaves ~2–10% (1 ms of 10–30 ms compute).
- Tokenizer / token-count calls are where FFI is meaningfully faster,
  because compute itself is sub-millisecond.

So the motivation is: *make integrations feasible*, not *make them fast*.

#### Scope

New crate at `llmserv/main/features/ffi/`:
- `crate-type = ["cdylib"]` — C-compatible ABI, so Python/Go/C#/Java/
  Swift consumers can load it via `ctypes`/`cgo`/`P/Invoke`/`JNI`/
  bridging headers. `dylib` (Rust ABI) is *not* an option — unstable
  across compiler versions.
- Narrow API surface. Don't expose the full inference library — expose
  what a desktop/IDE integration actually needs:
    * `llmserv_init(config_toml_path)` → opaque handle (loads model,
      applies quantization, constructs runtime per application.toml)
    * `llmserv_complete(handle, prompt, out_text)` → blocking completion
    * `llmserv_embed(handle, text, out_vec)` → blocking embedding
    * `llmserv_tokenize(handle, text, out_ids)` → tokenizer encode
    * `llmserv_token_count(handle, text)` → just the count (most common
      IDE call, every keystroke)
    * `llmserv_destroy(handle)`
    * `llmserv_free_*` — all Rust-allocated buffers freed via Rust
  Each function wrapped in `std::panic::catch_unwind`; panics never
  cross the FFI boundary.
- `cbindgen` auto-generates the C header; commit it alongside the
  `.so` output so consumers don't need a Rust toolchain to build bindings.

#### Priority 0 — minimum viable

**T2.1** — crate scaffold: Cargo.toml with `cdylib`, `#[no_mangle] extern
"C"` function stubs, `catch_unwind` wrappers, opaque handle type.
**T2.2** — core functions (init, complete, embed, destroy) backed by the
existing `rustml-model` / `rustml-generation` / `rustml-tokenizer`.
Blocking only — own a single-threaded tokio runtime inside the handle.
**T2.3** — `cbindgen.toml` and generated `include/llmserv.h`.
**T2.4** — Python smoke test (`ctypes` binding) that loads a model,
generates a few tokens, asserts the output is non-empty. Confirms the
ABI works end-to-end.

#### Priority 1 — add when a scenario demands

**T2.5** — streaming completion via a C callback:
`llmserv_complete_stream(handle, prompt, cb, cb_ctx)` where `cb` is
called per token. Needed for interactive IDE completion UIs.
**T2.6** — thread-safety contract documentation: can the same handle
be called from multiple caller threads concurrently? (Answer depends on
whether the inference stack allows re-entrant `generate`.)
**T2.7** — prebuilt `.so`/`.dll`/`.dylib` artifacts attached to GitHub
releases for each tagged version, so consumers don't need a Rust
toolchain to build llmserv.

#### Priority 2 — explicit non-goals (for now)

- **Async FFI.** FFI boundary is blocking; languages calling us can
  wrap in their own async (Python asyncio threadpool, Go goroutine).
- **Cross-version ABI stability guarantee.** Versioning via the soname
  / `.pc` / header version and bumping on every breaking surface change.
  Don't try to evolve `repr(C)` structs in place.
- **Multi-handle sharing of loaded weights.** Each handle owns its model.
  Could be added later via `Arc<Mmap>` shared between handles, but
  complexity isn't warranted until a real multi-handle consumer appears.
- **C++ classes / method syntax.** The `.h` is C, not C++. C++ callers
  include it in `extern "C" { ... }`.

#### Acceptance

- T2.1–T2.4 shipped: `cargo build --release -p llmserv-ffi` produces a
  `.dll` on Windows / `.so` on Linux / `.dylib` on macOS.
- `include/llmserv.h` committed and kept in sync with the Rust source
  (ideally regenerated in CI and diffed).
- Python smoke test script at `llmserv/main/features/ffi/examples/smoke.py`
  runs green against the built `.so`.
- An IDE integration or desktop consumer actually uses it — don't ship
  without a first consumer, it becomes dead code.

### T3: Migrate perf instrumentation to `swe-observ-tracing` + `tracing`

Replace the ad-hoc `log::debug!("[perf] foo::bar {:.3}ms", ...)` pattern
used across the inference stack with the standard `tracing` crate idioms,
backed by the sibling `swe-observ` workspace at
`C:/phd-systems/swelabs/observability/`.

**Why:** the current pattern works for what it is (we produced the P7.1
profile from it) but it's a reinvention. `tracing` + `tracing-subscriber`
is the Rust standard; `swe-observ-subscriber` provides an
`ObservabilityLayer: tracing_subscriber::Layer` that routes events and
spans to `swe-observ-tracing` / `swe-observ-logging` / `swe-observ-metrics`
pluggable backends (file, Jaeger, OTLP). Gains:

- Typed key-value span attributes (no regex parsing of format strings)
- Native histograms via `swe-observ-metrics` (no Python post-processing
  for percentiles)
- Pluggable backends including file JSON, Jaeger, OTLP
- Full `tracing` ecosystem compatible as additional layers:
  `tracing-flame` (flamegraphs when spans carry parent_span_id),
  `tokio-console` (async task inspector), `tracing-tree` / `tracing-forest`
  (dev pretty-print), `tracing-futures` (async-aware spans)
- Cross-project consistency with the `swe-observ` workspace

**Honest caveat about "unlocks visualization":** structured output does
NOT automatically give you flamegraphs. Flamegraphs need parent-child
span hierarchy. `swe-observ-tracing`'s SPI is `serde_json::Value` —
whatever JSON the caller emits is what gets exported. Hierarchy only
appears if we emit OTLP-shaped spans with `parent_span_id` fields (the
Jaeger/OTEL backends do this natively). Percentile histograms via
`swe-observ-metrics` ARE a real native win. Dashboards are marginally
easier with structured data but possible either way.

**Scope (~0.5 day):**
- Add workspace deps: `tracing`, `tracing-subscriber`,
  `swe-observ-subscriber` (via path or registry to the sibling workspace)
- Wire up subscriber at daemon startup (~10 lines in
  `llmserv/main/features/daemon/main/src/bin/serve.rs`):
  ```rust
  use tracing_subscriber::prelude::*;
  let observ = ObservabilityLayer::new(logging_backend, tracing_backend);
  tracing_subscriber::registry().with(observ).init();
  ```
- Mechanically convert ~30 `log::debug!("[perf] ...")` call sites:
  - Simple timers → `#[tracing::instrument(level = "debug")]` attribute
    (auto-timing, attrs from fn args)
  - Scoped blocks → `let _s = tracing::debug_span!("foo::bar", ...).entered();`
  - Metrics-style counters → `swe-observ-metrics` API directly
- Update the call-site list:
  - `generator.rs`: prefill, sample, decode_step (×2 paths: raw + chat)
  - `forward.rs`: embedding / layers / norm / projection
  - `attention.rs`, `feed_forward.rs`, `transformer_block.rs`: per-layer
  - `kv_cache.rs`: get_view, update
  - `linear.rs`, `quantize.rs`: per-op
- Re-run P7.1 profile with the new backend and confirm numbers match
  (embedding / layers / norm / projection percentages should be
  unchanged; that's the correctness check).

**Files touched (~12):**
- `Cargo.toml` (workspace deps)
- `llmserv/Cargo.toml` (workspace deps)
- `llmserv/main/features/daemon/main/src/bin/serve.rs` (subscriber init)
- All files with current `[perf]` debug log calls (mechanical rewrite)

**Explicit non-goals (for this T3):**
- Not writing our own Jaeger UI
- Not adding Prometheus metrics export (separate task if wanted)
- Not replacing every `log::info!` / `log::warn!` — those are fine as
  logs; only replace the `[perf]` timers

**Acceptance:**
- `RUST_LOG=rustml_generation=debug swellmd` still produces readable
  output on stderr (via `tracing-subscriber::fmt::Layer`).
- `ObservabilityLayer` wired to a file backend produces JSON Lines
  trace data at a configurable path.
- The P7.1 profile numbers (from `docs/` or the backlog) can be
  regenerated with the new instrumentation — a 2-page script runs a
  chat completion, parses the trace file, prints the same percentage
  breakdown we documented in P7.2 status.
- All existing tests pass; no behavior change visible to consumers.

**Risk:** low. `swe-observ-subscriber` already implements the `Layer`
trait; the `ObservabilityLayer` is just a glue layer. If anything goes
wrong, the fallback is `tracing_subscriber::fmt::Layer` for dev-grade
stderr output — we get the structured API without the swe-observ
backend path. Worst-case: we never wire the file backend and just use
the tracing ecosystem's own stdout formatter, which is still a win
over ad-hoc `log::debug!` parsing.

**Alternative (skip swe-observ, adopt plain tracing):** if the
sibling workspace isn't stable yet or we don't want the cross-workspace
coupling, we can migrate to just `tracing` + `tracing-subscriber` + a
standalone file or stdout appender. Smaller scope, same API surface in
the source code, just different backend plumbing. Preserves the option
to switch to `swe-observ-subscriber` later — source code stays
unchanged.

---

## Documentation Debt

### D1: Sweep docs for the CLI rename (sweai → llmc, rustml-cli → llm_cli)

In commit `8a293ec` (`feat(cli): rename to llmc; add 'load' subcommand`) the
developer CLI was renamed:
- crate:  `rustml-cli` → `llm_cli`
- binary: `sweai` → `llmc`

The code is consistent, but docs still reference the old names in **18 files**
(~194 total references).

**Rename mapping:**
- `sweai` → `llmc`
- `sweai infer` / `sweai hub` / `sweai gguf` / `sweai tokenizer` — subcommand names unchanged; just swap `sweai` → `llmc`
- `cargo build -p rustml-cli` → `cargo build -p llm_cli`
- `-p rustml-cli` (as a cargo package selector) → `-p llm_cli`

**Files, ordered by reference count** (biggest offenders first):

| File | Refs |
|---|---|
| `docs/5-testing/manual_sweai_tests.md` | 81 |
| `docs/7-operations/user_manual.md` | 54 |
| `docs/5-testing/manual_testing.md` | 20 |
| `docs/5-testing/manual_infer_tests.md` | 7 |
| `docs/4-development/developer_guide.md` | 6 |
| `docs/6-deployment/deployment_guide.md` | 4 |
| `llmserv/main/features/experimentation/llmforge/ARCHIVED.md` | 3 |
| `docs/3-design/adr/adr-001-unified-llmmodel-for-gpt2.md` | 2 |
| `docs/3-design/adr/adr-002-retire-llmforge-prototype.md` | 2 |
| `docs/3-design/inference_dataflow.md` | 2 |
| `docs/3-design/project_structure.md` | 2 |
| `docs/5-testing/manual_gguf_inspect_tests.md` | 2 |
| `docs/5-testing/manual_hub_cli_tests.md` | 2 |
| `docs/5-testing/manual_tokenizer_tests.md` | 2 |
| `docs/7-operations/operations_guide.md` | 2 |
| `docs/4-development/guides/model-verification.md` | 1 |
| `llmserv/README.md` | 1 |
| `main/features/tensor/docs/0-ideation/growth-roadmap.md` | 1 |

**Additional rename to apply at the same time:** `manual_sweai_tests.md` itself
should be renamed to `manual_llmc_tests.md` (the file name is now stale).

**Out of scope:** ADRs are historical records. Consider leaving them alone,
or prefixing with a "nomenclature note" that says what the old names are now
called, rather than rewriting history.

**Acceptance:**
- `grep -rn "sweai\|rustml-cli" docs/ llmserv/ main/ --include="*.md"` returns
  only historical ADR references (if any).
- `manual_sweai_tests.md` renamed.
- `git grep -n "\`sweai\`\|\`rustml-cli\`"` is clean in active docs.

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
