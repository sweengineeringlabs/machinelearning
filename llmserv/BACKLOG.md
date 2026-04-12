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

### P6: GPU acceleration (Vulkan)
- **Architecture done**: `rustml-compute` crate with `ComputeBackend` trait, `CpuBackend` provider (wraps existing ops), `VulkanBackend` stub
- **Remaining**: Implement Vulkan compute shaders for matmul, softmax, GELU, SiLU. Buffer management, shader compilation pipeline, device selection. Wire `ComputeBackend` into `inference/layers/` to replace CPU tensor ops.
- **Expected impact**: 10-50x speedup for matmul-bound workloads
- **Complexity**: Major project

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
