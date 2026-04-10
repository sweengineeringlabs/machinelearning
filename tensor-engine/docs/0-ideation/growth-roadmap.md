# Growth Roadmap: rustml Framework

**Date:** 2026-04-10
**Scope:** Unified roadmap across rustml (training framework) and machinelearning (inference engine) — covering tensor engine, model ecosystem, and deployment targets.
**Ref:** [Market Research](market-research.md) | [SRS](../../timeseries/docs/1-requirements/srs.md) | [Model Backlog](../../../../rustml/rustml-hub/doc/2-planning/backlog.md) | [Inference Backlog](../../docs/4-development/backlog.md)

---

## 1. Current State

### What works today

| Layer | Status | Key assets |
|-------|--------|------------|
| **Tensor engine** | Production-usable | Multi-dtype (F32/F16/BF16/INT8/Q4), SIMD kernels (AVX2/NEON), Arc-based storage |
| **Autograd** | Complete | Tape-based reverse-mode AD, all core ops have backward implementations |
| **NN layers** | Complete | Linear, MultiHeadAttention, FeedForward, RMSNorm, LayerNorm, BatchNorm, RoPE, MoE, Embedding |
| **Activations** | Mostly complete | ReLU (+ in-place), GELU, SiLU. Missing: sigmoid, tanh |
| **Optimizers** | Complete | SGD, AdamW, LR schedulers, gradient clipping |
| **Training** | Complete | Trainer with early stopping, checkpointing, metrics, hyperparameter tuning |
| **LLM inference** | Production-usable | GPT-2, BERT, GLM-4 via SafeTensors + GGUF. KV-cache, Q8_0 quantization, fused QKV/gate-up |
| **Time series** | In progress | PatchTST, Informer, N-BEATS, DLinear, TCN, LSTM. Transformer training validated |
| **Model loading** | Working | HuggingFace Hub download, SafeTensors, GGUF, weight mapping |
| **Perf baseline** | Measured | GPT-2: 9.3 tok/s (SafeTensors), 2.07 tok/s (large models). Attention 1.8x faster than ndarray |

### What's missing

| Gap | Impact | Documented? |
|-----|--------|-------------|
| GPU compute | 10-50x inference speed ceiling | Mentioned in backlog, WGPU shader exists but untested at scale |
| ONNX import | Can't load models from PyTorch/TF ecosystem | ADR-006 exists, not implemented |
| WASM target | No browser/serverless deployment | Not documented |
| sigmoid/tanh activations | `todo!()` panics in production | Not documented |
| Flash/GQA attention | Can't run modern LLMs efficiently | Not documented |
| TensorPool buffer recycling | Allocation pressure during backward pass | Spec'd in SRS (FR-104), not implemented |
| Cross-repo alignment | rustml and machinelearning evolve independently | Not documented |
| Unified roadmap | No 6-12 month plan | This document |

---

## 2. Strategic Direction

### Thesis

Rust ML is strongest where Python is weakest: edge deployment, safety-critical systems, deterministic latency, single-binary distribution. The growth strategy is:

1. **Time series as beachhead** — no Rust competitor exists, win the vertical
2. **LLM inference as growth vector** — large community demand, proven by llama.cpp
3. **Pure Rust full-stack as moat** — train, quantize, deploy in one language with zero C/C++ dependencies

### Non-goals

- Competing with PyTorch for research training workflows
- Supporting every model architecture
- GPU training (CPU training for small models is the sweet spot)

---

## 3. Roadmap

### Phase 1: Foundations (Q2 2026)

Complete the gaps that block production use and close known regressions.

| ID | Item | Priority | Effort | Crate | Description |
|----|------|----------|--------|-------|-------------|
| R-001 | sigmoid and tanh activations | Must | 1 day | rustml-core | Replace `todo!()` with implementations + autograd backward. Blocks any model using these activations |
| R-002 | In-place GELU and SiLU | Should | 1 day | rustml-core | Same pattern as `relu_inplace` — `Arc::make_mut` + `mapv_inplace` with autograd guard |
| R-003 | TensorPool for backward pass | Should | 3 days | rustml-core | Buffer recycling during backward (SRS FR-104). Reduces allocation pressure in training loops |
| R-004 | LayerNorm in transformer encoder | Should | 1 day | rustml-nn | TransformerEncoderLayer uses BatchNorm1d — should be LayerNorm for standard transformer behavior |
| R-005 | Benchmark suite | Must | 2 days | rustml-core | Criterion benchmarks covering matmul, attention, relu, softmax at multiple sizes. Regression detection |
| R-006 | Fix `bench_all_optimizations` blocking tests | Must | 1 hour | machinelearning | Add `#[ignore]` to the 76-minute benchmark test (documented in backlog) |

**Exit criteria:** All `todo!()` activations implemented. Benchmark suite running in CI. No known panics in production paths.

### Phase 2: Model Ecosystem (Q3 2026)

Expand the model zoo to cover the models people actually run.

| ID | Item | Priority | Effort | Crate | Description |
|----|------|----------|--------|-------|-------------|
| R-010 | GGUF model loading | Must | 1 week | machinelearning | Already supported in inference engine. Expose as a clean API for rustml-hub |
| R-011 | LLaMA architecture | Must | 2 weeks | rustml-nlp | The most-requested open model family. GQA, RoPE (already have RoPE), SiLU FFN |
| R-012 | Mistral / Phi support | Should | 1 week | rustml-nlp | Sliding window attention (Mistral), shared embeddings (Phi). Builds on LLaMA arch |
| R-013 | ONNX import | Should | 2 weeks | rustml-hub | Load ONNX graphs for inference. Use `ort` crate or implement parser. ADR-006 exists |
| R-014 | Time series foundation models | Should | 3 weeks | rustml-timeseries | Chronos or TimesFM — pretrained forecasting models. Major differentiator |
| R-015 | Autoformer + iTransformer | Should | 2 weeks | rustml-timeseries | Top-performing time series architectures. TS-002 and TS-005 in backlog |
| R-016 | HubPortable trait | Should | 3 days | rustml-hub | Standardize model loading interface (MP-010 in backlog). Unblocks model registry |
| R-017 | Model registry | Nice | 2 days | rustml-hub | Discover supported models programmatically (MP-011 in backlog) |

**Exit criteria:** LLaMA 7B/8B runs in rustml with GGUF weights. At least one time series foundation model operational.

### Phase 3: Performance (Q3-Q4 2026)

Close the performance gap on operations that matter.

| ID | Item | Priority | Effort | Crate | Description |
|----|------|----------|--------|-------|-------------|
| R-020 | Fused attention kernel | Must | 1 week | rustml-core | Q@K→scale→mask→softmax→@V as single pass. Expected 2-3x on attention-heavy workloads |
| R-021 | Flash Attention | Should | 2 weeks | rustml-core | Tiled attention with O(N) memory instead of O(N^2). Required for long sequences (>2K) |
| R-022 | Grouped-Query Attention (GQA) | Must | 3 days | rustml-nn | Required for LLaMA 2/3, Mistral. Fewer KV heads than Q heads |
| R-023 | Zero-copy reshape/slice | Should | 1 week | rustml-core | Avoid allocation for shape-only ops. Careful: non-contiguous layouts slower for BLAS (see backlog key finding) |
| R-024 | Parallel data loading | Should | 1 week | rustml-data | Async prefetch with rayon. Training currently blocks on data |
| R-025 | WGPU matmul integration | Nice | 2 weeks | rustml-core | GPU-accelerated matmul via compute shaders. WGPU shader exists in machinelearning, needs porting |

**Exit criteria:** Attention 2x faster at seq=512+. GQA implemented and validated with LLaMA weights.

### Phase 4: Deployment Targets (Q4 2026 - Q1 2027)

Expand where rustml models can run.

| ID | Item | Priority | Effort | Crate | Description |
|----|------|----------|--------|-------|-------------|
| R-030 | WASM compilation target | Must | 2 weeks | rustml-core | Compile tensor engine + inference to WASM. candle and tract prove this works |
| R-031 | `no_std` core subset | Should | 2 weeks | rustml-core | Enable embedded/bare-metal deployment. Requires removing std dependencies from hot path |
| R-032 | Static binary packaging | Should | 3 days | rustml-cli | Single binary with model weights embedded. `include_bytes!()` or appended payload |
| R-033 | C FFI bindings | Nice | 1 week | rustml-core | Enable calling rustml from C/C++/Python/Swift. Expands adoption surface |
| R-034 | Python bindings (PyO3) | Nice | 2 weeks | new crate | `pip install rustml` — lowers adoption barrier. Train in Python, deploy in Rust |
| R-035 | safetensors export | Should | 3 days | rustml-hub | Save trained models in safetensors format. Enables rustml→HuggingFace pipeline |

**Exit criteria:** rustml model runs in browser via WASM. At least one model packaged as single static binary.

### Phase 5: Ecosystem Maturity (2027+)

Long-term investments for community adoption.

| ID | Item | Priority | Effort | Description |
|----|------|----------|--------|-------------|
| R-040 | Distributed training | Nice | 3 months | Multi-machine training via gRPC/MPI. Only if demand emerges |
| R-041 | Model marketplace | Nice | 1 month | Community-contributed models with versioning and benchmarks |
| R-042 | Visualization / TensorBoard | Nice | 2 weeks | Training metrics export to TensorBoard or custom web UI |
| R-043 | Differential privacy | Nice | 3 weeks | DP-SGD for privacy-preserving training. Regulatory advantage |
| R-044 | Federated learning | Nice | 2 months | Train across devices without centralizing data |

---

## 4. Cross-Repository Alignment

### Current problem

Two repositories evolve independently:

| Repo | Focus | Tensor impl |
|------|-------|-------------|
| `rustml` | Training framework, model zoo, HuggingFace hub | ndarray-backed `rustml-core::Tensor` |
| `machinelearning` | LLM inference, quantization, SIMD | Custom `tensor-engine` with multi-dtype storage |

### Alignment plan

| Step | Action | When |
|------|--------|------|
| 1 | Unify tensor types — adopt `rustml-core::Tensor` as the single tensor type, with `tensor-engine` SIMD kernels as the backend | Phase 1-2 |
| 2 | Merge model implementations — LLM models in `machinelearning/llm` should use `rustml-nn` layers | Phase 2 |
| 3 | Single workspace — combine into one Cargo workspace or use path dependencies with clear API boundaries | Phase 3 |
| 4 | Shared CI — unified test suite, benchmarks, and release process | Phase 3 |

---

## 5. Competitive Moat Priorities

Ranked by defensibility and market impact:

| Rank | Moat | Why it matters | Timeline |
|------|------|---------------|----------|
| 1 | **Time series in Rust** | Zero competitors. Greenfield vertical | Now — Q3 2026 |
| 2 | **Full-stack Rust ML** | Train → quantize → deploy, no Python. burn is close but lacks model zoo | Q2 — Q4 2026 |
| 3 | **GGUF/safetensors interop** | Access the entire HuggingFace + llama.cpp ecosystem without reimplementing | Q2 — Q3 2026 |
| 4 | **Edge training** | tract and candle are inference-only. On-device fine-tuning/online learning has no Rust solution | Q4 2026 |
| 5 | **WASM deployment** | Browser inference is underserved. candle does it but is inference-only | Q4 2026 |
| 6 | **Safety-critical ML** | Rust's memory safety is a regulatory advantage in automotive/medical/aerospace | 2027+ |

---

## 6. Success Metrics

| Metric | Current | Phase 2 target | Phase 4 target |
|--------|---------|---------------|----------------|
| Models supported | 8 (GPT-2, BERT, GLM-4, PatchTST, Informer, N-BEATS, DLinear, TCN) | 12+ (add LLaMA, Mistral, Chronos, Autoformer) | 15+ |
| GPT-2 inference | 9.3 tok/s | 9.3 tok/s (maintain) | 15+ tok/s (WGPU) |
| LLaMA 8B inference | N/A | 5+ tok/s (Q8, CPU) | 20+ tok/s (GPU) |
| Attention perf vs ndarray | 1.8x faster | 3x faster (fused) | 5x faster (Flash) |
| Deployment targets | CPU binary | + WASM | + embedded, C FFI, Python |
| Test coverage | ~93 tests (core + nn) | 150+ | 300+ |
| `todo!()` in production paths | 2 (sigmoid, tanh) | 0 | 0 |

---

## 7. Dependencies and Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| burn ships LLaMA + time series before us | Medium | High | Prioritize LLaMA (R-011) and time series models (R-014, R-015) in Phase 2 |
| candle adds training support | Low | Medium | Our model zoo and time series specialization are durable differentiators |
| Two-repo divergence increases maintenance burden | High | Medium | Unify tensor types in Phase 1-2 (Section 4) |
| WGPU matmul doesn't match CUDA perf | Medium | Low | CPU-first positioning means GPU is additive, not required |
| Quantized model quality degrades at Q4 | Medium | Medium | Validate perplexity at each quantization level before shipping |
| In-place ops bypass autograd (SRS 5.6) | Low | High | Runtime guards implemented. New in-place ops must use `map_inplace` |

---

## Appendix: ID Cross-Reference

Maps roadmap items to existing backlog entries:

| Roadmap ID | Backlog ID | Source |
|------------|-----------|--------|
| R-006 | bench_all_optimizations | machinelearning backlog |
| R-010 | — | GGUF already works in machinelearning/llm |
| R-013 | ADR-006, MP-012 | rustml-hub backlog |
| R-015 | TS-002, TS-005 | rustml-hub backlog |
| R-016 | MP-010 | rustml-hub backlog |
| R-017 | MP-011 | rustml-hub backlog |
| R-023 | MP-006 | rustml-hub backlog (blocked — see key finding on BLAS) |
| R-025 | MP-015 | rustml-hub backlog (WGPU shader exists) |
