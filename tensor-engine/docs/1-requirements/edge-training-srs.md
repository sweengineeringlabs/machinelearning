# Software Requirements Specification: Road to Edge Training

**Audience:** Developers, architects, contributors

**Project:** rustml-edge — On-device ML training in pure Rust, from Linux edge to bare metal.

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Status | Draft |
| Date | 2026-04-10 |
| Parent | `rustml` umbrella workspace |
| Ref | [Growth Roadmap](../0-ideation/growth-roadmap.md), [Market Research](../0-ideation/market-research.md) |

---

## 1. Purpose

The ML industry is shifting from cloud-only training to **on-device training** — models that learn where the data lives, because the data cannot leave:

| Domain | Constraint | Example |
|--------|-----------|---------|
| Consumer devices | Privacy | Apple on-device personalization (Siri, keyboard, photos) |
| Mobile fleet | Scale | Google federated learning across 3B Android devices |
| Automotive | Bandwidth | Tesla — cameras generate TB/day per vehicle |
| Medical devices | Regulation | HIPAA/GDPR prohibit patient data leaving the device |
| Industrial IoT | Latency | Predictive maintenance models retraining on local equipment drift |

**The tooling gap:** Python cannot run on constrained devices (no runtime, no GC tolerance). C++ can but is not memory-safe (unacceptable in medical/automotive/aerospace). Every existing Rust ML framework (candle, tract, ort) is inference-only.

**rustml-edge** delivers on-device training in pure Rust — memory-safe, no runtime, deterministic latency, with the ability to fine-tune and train at the edge.

### 1.1 Scope

This SRS covers the requirements to bring rustml from its current state (desktop/server training) to edge deployment across two tiers:

| Tier | Target | OS | Example hardware | Priority |
|------|--------|----|--------------------|----------|
| **Tier 1** | Linux edge | Linux (std available) | Raspberry Pi 4/5, NVIDIA Jetson, Industrial PCs, Medical devices on embedded Linux | Must |
| **Tier 2** | Bare metal | RTOS or none (no_std + alloc) | ARM Cortex-M, RISC-V, custom FPGA SoCs | Should |

### 1.2 Relationship to Existing Crates

```
                    ┌─────────────────────────────┐
                    │        rustml-nn             │
                    │  (layers, attention, MoE)    │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    TensorBackend trait       │
                    │  matmul, relu, softmax, ...  │
                    └──────┬───────────┬──────────┘
                           │           │
              ┌────────────▼──┐  ┌─────▼────────────┐
              │  rustml-core  │  │  tensor-engine    │
              │  (ndarray)    │  │  (custom, no_std) │
              │  std, rayon   │  │  SIMD, alloc-only │
              │  server/desk  │  │  edge/embedded    │
              └───────────────┘  └───────────────────┘
```

| Component | Current state | Edge role |
|-----------|--------------|-----------|
| `rustml-core` | ndarray-backed tensor with autograd | Tier 1 backend (Linux edge, cross-compiled to ARM) |
| `tensor-engine` | Custom multi-dtype tensor, SIMD kernels, no autograd | Tier 2 backend candidate (no ndarray dependency) |
| `rustml-nn` | Layers hardcoded to `rustml-core::Tensor` | Must be generic over backend trait |
| `rustml-train` | Trainer, optimizers, schedulers | Must work with bounded memory on edge |

---

## 2. Functional Requirements

### 2.1 Tier 1 — Linux Edge (Cross-Compilation)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-100 | rustml-core compiles for `aarch64-unknown-linux-gnu` (ARM64 Linux) | Must |
| FR-101 | rustml-core compiles for `armv7-unknown-linux-gnueabihf` (ARM32 Linux) | Should |
| FR-102 | Full training loop (forward, backward, optimizer step) executes on ARM Linux | Must |
| FR-103 | NEON SIMD kernels activate automatically on ARM targets | Must |
| FR-104 | Pre-trained model loads from file on edge device (safetensors or rustml format) | Must |
| FR-105 | Fine-tuned model saves to file on edge device | Must |
| FR-106 | Training runs within a fixed memory budget (pre-allocated, no unbounded growth) | Must |
| FR-107 | Binary size under 50 MB for a complete training application (stripped, LTO) | Should |
| FR-108 | Binary size under 20 MB for inference-only application | Should |

### 2.2 Tier 2 — Bare Metal (no_std)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-200 | A `no_std + alloc` tensor backend exists with: matmul, add, mul, relu, softmax, transpose | Must |
| FR-201 | Autograd (gradient tape + backward pass) works on `no_std` backend | Must |
| FR-202 | At least one optimizer (SGD) works on `no_std` backend | Must |
| FR-203 | Forward + backward pass executes with zero heap allocation after initialization | Should |
| FR-204 | All tensor buffers pre-allocatable via arena/pool at startup | Should |
| FR-205 | Compiles for `thumbv7em-none-eabihf` (Cortex-M4/M7) | Should |
| FR-206 | Compiles for `riscv32imc-unknown-none-elf` (RISC-V) | Nice |

### 2.3 TensorBackend Trait

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-300 | `TensorBackend` trait abstracts tensor creation, math ops, and memory layout | Must |
| FR-301 | `rustml-core::Tensor` implements `TensorBackend` (ndarray backend) | Must |
| FR-302 | `tensor-engine::Tensor` implements `TensorBackend` (custom backend) | Must |
| FR-303 | `rustml-nn` layers are generic over `TensorBackend`, not hardcoded to a concrete tensor type | Must |
| FR-304 | Autograd tape is generic over `TensorBackend` | Must |
| FR-305 | Backend selection is compile-time (feature flags), not runtime dispatch | Must |
| FR-306 | `TensorBackend` trait is object-safe where possible, but static dispatch (`impl TensorBackend`) is preferred for performance | Should |

### 2.4 Memory Management

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-400 | `TensorPool` provides pre-allocated buffer recycling for forward and backward passes | Must |
| FR-401 | `TensorPool` has a configurable memory ceiling (e.g., 256 MB) | Must |
| FR-402 | Training loop reports peak memory usage after each epoch | Should |
| FR-403 | OOM condition is a recoverable error, not a panic | Must |
| FR-404 | Gradient accumulation supports micro-batching to reduce peak memory (trade compute for memory) | Should |
| FR-405 | Model weights can be memory-mapped from flash/disk without copying into RAM | Should |

### 2.5 On-Device Fine-Tuning

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-500 | Load a pre-trained model, freeze base layers, fine-tune head on local data | Must |
| FR-501 | Layer freezing: `layer.freeze()` excludes parameters from gradient computation and optimizer updates | Must |
| FR-502 | LoRA (Low-Rank Adaptation): inject trainable low-rank matrices into frozen layers | Should |
| FR-503 | Quantization-aware training: forward pass in INT8, gradient computation in FP32 | Should |
| FR-504 | Online learning: update model weights continuously from a data stream without epoch boundaries | Should |
| FR-505 | Checkpoint to persistent storage after N updates (crash recovery) | Must |
| FR-506 | Differential checkpoint: save only changed parameters, not full model | Should |

### 2.6 Federated Learning Primitives

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-600 | Extract model gradients or weight deltas as serializable payload | Should |
| FR-601 | Apply aggregated gradients/deltas received from coordinator | Should |
| FR-602 | Differential privacy: clip per-sample gradients and add calibrated noise | Nice |
| FR-603 | Secure aggregation: encrypt gradients before transmission | Nice |

---

## 3. Non-Functional Requirements

### 3.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-100 | Training throughput on Raspberry Pi 4 (Cortex-A72, 4GB RAM) | > 10 samples/sec for 1M-param model |
| NFR-101 | Training throughput on Jetson Nano (Maxwell GPU, 4GB RAM) | > 100 samples/sec for 1M-param model (CPU path) |
| NFR-102 | Inference latency on Cortex-M7 (480 MHz, 1MB RAM) | < 100ms for 100K-param model |
| NFR-103 | NEON SIMD utilization on ARM64 | Matmul within 2x of OpenBLAS single-threaded |
| NFR-104 | Peak memory during training | < 4x model size (weights + gradients + activations + optimizer state) |
| NFR-105 | Startup time (model load to first inference) | < 2 seconds on Tier 1, < 500ms on Tier 2 |

### 3.2 Binary Size

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-200 | Stripped release binary (training + model + runtime) | < 50 MB (Tier 1) |
| NFR-201 | Stripped release binary (inference only) | < 20 MB (Tier 1) |
| NFR-202 | Compiled firmware (no_std, inference only) | < 2 MB (Tier 2) |
| NFR-203 | Compiled firmware (no_std, training + inference) | < 5 MB (Tier 2) |
| NFR-204 | No dynamic linking — fully static binary | All tiers |

### 3.3 Correctness

| ID | Requirement |
|----|-------------|
| NFR-300 | Edge-trained model produces identical gradients to desktop-trained model given same input and weights (bitwise for FP32, within tolerance for quantized) |
| NFR-301 | Fine-tuned model checkpoint loads correctly on both edge and desktop |
| NFR-302 | TensorPool buffer recycling does not corrupt gradient computation |
| NFR-303 | Memory ceiling enforcement does not silently drop operations — fails explicitly |

### 3.4 Safety

| ID | Requirement |
|----|-------------|
| NFR-400 | No `unsafe` code outside SIMD kernels and FFI boundaries |
| NFR-401 | No undefined behavior on any target (verified by Miri where applicable) |
| NFR-402 | Stack usage bounded and documented for `no_std` targets (no recursive algorithms without depth limit) |
| NFR-403 | No panic in library code on OOM — return `Result::Err` |
| NFR-404 | Thread safety: all shared state protected by appropriate synchronization primitives |

### 3.5 Compatibility

| ID | Requirement |
|----|-------------|
| NFR-500 | Minimum supported Rust version: stable (no nightly-only features in default build) |
| NFR-501 | SIMD intrinsics behind feature gates (`neon`, `avx2`) with scalar fallback |
| NFR-502 | Model format interoperable between edge and desktop builds |
| NFR-503 | `TensorBackend` trait compatible with both `std` and `no_std + alloc` environments |
| NFR-504 | Weights trained on desktop (FP32) loadable on edge (with optional quantization at load time) |

### 3.6 Extensibility

| ID | Requirement |
|----|-------------|
| NFR-600 | New `TensorBackend` implementations can be added without modifying `rustml-nn` or `rustml-train` |
| NFR-601 | New target architectures require only a cross-compilation target, not code changes (Tier 1) |
| NFR-602 | Custom SIMD kernels can be registered per-backend without modifying core trait |

---

## 4. Constraints

| ID | Constraint |
|----|-----------|
| CON-01 | Tier 1 is CPU-only — no GPU on edge Linux (Jetson GPU is out of scope for v1) |
| CON-02 | Tier 2 targets single-core execution — no threading primitives available |
| CON-03 | No dynamic memory allocation after initialization on Tier 2 (arena/pool only) |
| CON-04 | Model size limited by device RAM — no disk-based swapping on Tier 2 |
| CON-05 | No network stack assumed on Tier 2 — federated learning requires Tier 1 |
| CON-06 | Pure Rust — no C/C++ dependencies (ndarray is Rust, matrixmultiply is Rust) |
| CON-07 | `tensor-engine` is the Tier 2 backend; `rustml-core` (ndarray) is not viable for `no_std` |

---

## 5. Known Design Risks

### 5.1 TensorBackend Trait Explosion (Critical)

**Problem:** Making `rustml-nn` generic over a `TensorBackend` trait requires every layer, every activation, and every loss function to become generic. This can lead to trait bound explosion (`where T: TensorBackend + Clone + Send + Sync + ...`) and significant refactoring of the entire codebase.

**Impact:** Could double the complexity of `rustml-nn` and make the code harder to read and maintain.

**Mitigation options:**
1. **Type alias approach:** `type Tensor = rustml_core::Tensor` or `type Tensor = tensor_engine::Tensor` selected by feature flag. No generics, just conditional compilation. Simplest but only supports one backend per binary.
2. **Trait approach with associated types:** `trait TensorBackend { type Tensor: TensorOps; }` — cleaner bounds but requires careful design.
3. **Enum dispatch:** Single `Tensor` enum wrapping both backends. Runtime overhead but no generic proliferation.

**Recommendation:** Start with option 1 (type alias + feature flag). It delivers Tier 1 and Tier 2 without refactoring the entire NN stack. Migrate to option 2 only if multi-backend-in-same-binary becomes a real requirement.

### 5.2 Autograd on no_std (High)

**Problem:** The current autograd system uses `thread_local!`, `Mutex`, `OnceLock`, and `HashMap` — all requiring `std`. The `tensor-engine` has no autograd at all.

**Impact:** Tier 2 training is blocked until autograd works without `std`.

**Mitigation:**
1. Replace `thread_local!` with a global static behind `spin::Mutex` (single-core on Tier 2, so no contention)
2. Replace `HashMap` with `BTreeMap` (available in `alloc`)
3. Replace `OnceLock` with `spin::Once` or `once_cell` (supports `no_std`)
4. Port autograd to `tensor-engine` as a separate module, using only `core` + `alloc`

### 5.3 ndarray Binary Size (Medium)

**Problem:** ndarray pulls in rayon, crossbeam, and other threading infrastructure. Even for Tier 1 (Linux edge), this may bloat the binary beyond the 50 MB target.

**Impact:** May fail NFR-200 binary size requirement.

**Mitigation:**
1. Measure actual binary size after `strip` + LTO before optimizing
2. Disable rayon feature on ndarray for edge builds (accept single-threaded)
3. If still too large, use `tensor-engine` for Tier 1 as well (eliminates ndarray entirely)

### 5.4 NEON SIMD Coverage (Medium)

**Problem:** SIMD kernels in `rustml-core/src/simd_ops.rs` and `tensor-engine` target AVX2 (x86). NEON (ARM) coverage may be incomplete.

**Impact:** Tier 1 ARM targets run scalar fallbacks, missing NFR-103 performance target.

**Mitigation:**
1. Audit existing NEON kernels in tensor-engine (`rms_norm_row_neon`, `dot_q8_block_neon`)
2. Add NEON matmul kernel if missing
3. Use `matrixmultiply` crate (pure Rust, auto-vectorizes for ARM) as fallback

### 5.5 TensorPool Correctness Under Bounded Memory (Medium)

**Problem:** A fixed-size TensorPool that recycles buffers must correctly handle the case where backward pass requires more simultaneous buffers than the pool can provide. Unlike desktop (where malloc always succeeds until OOM), the pool has a hard ceiling.

**Impact:** Training silently produces wrong gradients if buffers are reused while still live, or panics if pool is exhausted.

**Mitigation:**
1. Reference counting on pooled buffers — buffer is not recyclable until all references are dropped
2. Explicit `Result::Err` when pool is exhausted (not panic)
3. Pool size calculator: given model architecture, compute minimum pool size needed for one forward+backward pass
4. Validate pool correctness via gradient finite-difference checks (same as NFR-300)

### 5.6 In-Place Operations Bypass Autograd

**Problem:** In-place mutation methods (`relu_inplace`, `map_inplace`) do not record operations on the gradient tape. If called on a tensor in the training graph, gradients will be silently incorrect.

**Impact:** Silent gradient corruption during training.

**Mitigation (implemented):** Runtime assert that panics if called on grad-tracked tensor while tape is active. See swets SRS section 5.6 for details.

**Residual risk:** Runtime-only guard. New in-place methods must include the same guard.

---

## 6. Implementation Phases

### Phase 1: Prove Tier 1 (2 weeks)

Validate that the current codebase works on Linux edge without architectural changes.

| Step | Deliverable | Effort |
|------|------------|--------|
| 1.1 | Cross-compile rustml to `aarch64-unknown-linux-gnu` | 1 day |
| 1.2 | Run full test suite on ARM Linux (Pi 4 or QEMU) | 1 day |
| 1.3 | Measure binary size (stripped, LTO, with/without rayon) | 1 day |
| 1.4 | Benchmark: train 1M-param time series model on Pi 4, measure throughput and peak memory | 2 days |
| 1.5 | Demo application: load pretrained PatchTST, fine-tune on local CSV, save checkpoint | 3 days |
| 1.6 | Document results: binary size, throughput, memory, any failures | 1 day |

**Exit criteria:** Training loop runs on ARM Linux. Binary under 50 MB. Throughput and memory documented.

### Phase 2: TensorBackend Abstraction (3 weeks)

Decouple `rustml-nn` from concrete tensor type to enable backend swapping.

| Step | Deliverable | Effort |
|------|------------|--------|
| 2.1 | Define `TensorBackend` type alias in `rustml-core` (feature-flag approach, Risk 5.1 option 1) | 2 days |
| 2.2 | Audit all `rustml-nn` layers for direct ndarray usage — replace with `Tensor` methods | 3 days |
| 2.3 | Implement `TensorPool` with configurable memory ceiling (FR-400, FR-401) | 3 days |
| 2.4 | Add memory reporting to training loop (FR-402) | 1 day |
| 2.5 | Validate: full test suite passes with type alias pointing to ndarray backend | 2 days |
| 2.6 | Measure: peak memory during PatchTST training, verify < 4x model size | 1 day |

**Exit criteria:** `rustml-nn` has no direct ndarray imports. TensorPool operational. Memory ceiling enforced.

### Phase 3: Autograd on tensor-engine (3 weeks)

Port gradient tracking to the custom tensor engine for `no_std` viability.

| Step | Deliverable | Effort |
|------|------------|--------|
| 3.1 | Port `GradientTape` to tensor-engine using `alloc`-only primitives (BTreeMap, spin::Mutex) | 5 days |
| 3.2 | Implement backward ops for: matmul, add, mul, relu, softmax, transpose | 5 days |
| 3.3 | Implement SGD optimizer on tensor-engine tensors | 2 days |
| 3.4 | Gradient correctness: finite-difference checks for all backward ops | 2 days |
| 3.5 | Validate: train a 2-layer MLP on tensor-engine, verify loss convergence | 1 day |

**Exit criteria:** tensor-engine has autograd. Gradients match rustml-core within FP32 tolerance. Simple model trains to convergence.

### Phase 4: no_std Compilation (2 weeks)

Feature-gate tensor-engine for bare metal targets.

| Step | Deliverable | Effort |
|------|------------|--------|
| 4.1 | Add `#![cfg_attr(not(feature = "std"), no_std)]` to tensor-engine | 1 day |
| 4.2 | Replace `std` dependencies: HashMap → BTreeMap, Mutex → spin::Mutex, OnceLock → spin::Once | 3 days |
| 4.3 | Feature-gate `rand` (random init not available on no_std — require explicit weight loading) | 1 day |
| 4.4 | Feature-gate file I/O (checkpoint save/load behind `std` feature) | 1 day |
| 4.5 | Compile for `thumbv7em-none-eabihf` — resolve all compilation errors | 2 days |
| 4.6 | Measure firmware size (inference-only and training) | 1 day |
| 4.7 | Run inference on Cortex-M7 (real hardware or QEMU) | 1 day |

**Exit criteria:** tensor-engine compiles for Cortex-M. Firmware under 5 MB. Inference executes on target.

### Phase 5: Edge Training Demo (2 weeks)

End-to-end demonstration of on-device training.

| Step | Deliverable | Effort |
|------|------------|--------|
| 5.1 | Tier 1 demo: Fine-tune time series model on Raspberry Pi from local sensor CSV | 3 days |
| 5.2 | Tier 2 demo: Train 100K-param MLP on Cortex-M7 from embedded data | 3 days |
| 5.3 | Benchmark report: throughput, memory, binary size, power consumption (if measurable) | 2 days |
| 5.4 | Documentation: "Getting Started with Edge Training" guide | 2 days |

**Exit criteria:** Both tier demos run successfully. Results documented with reproducible steps.

---

## 7. Acceptance Criteria

### 7.1 Tier 1 Acceptance

| Test | Criteria |
|------|----------|
| Cross-compilation | `cargo build --target aarch64-unknown-linux-gnu` succeeds with zero errors |
| Training correctness | Same model + data + seed produces identical loss curve on ARM and x86 (FP32) |
| Binary size | Stripped training binary < 50 MB |
| Throughput | > 10 samples/sec for 1M-param model on Raspberry Pi 4 |
| Memory | Peak memory < 4x model size during training |
| Checkpoint round-trip | Model fine-tuned on edge loads and produces identical output on desktop |

### 7.2 Tier 2 Acceptance

| Test | Criteria |
|------|----------|
| Compilation | `cargo build --target thumbv7em-none-eabihf` succeeds with `no_std` |
| Firmware size | Training firmware < 5 MB, inference firmware < 2 MB |
| Inference correctness | Output matches desktop inference within FP32 tolerance |
| Training convergence | 100K-param model loss decreases over 100 iterations on target |
| Zero alloc after init | No heap allocation during forward+backward pass (verified by custom allocator) |
| Gradient correctness | Finite-difference check passes for all backward ops on target |

### 7.3 TensorBackend Acceptance

| Test | Criteria |
|------|----------|
| Backend swap | Changing feature flag from `backend-ndarray` to `backend-tensor-engine` compiles and passes all `rustml-nn` tests |
| No ndarray leakage | `grep -r "ndarray" rustml-nn/src/` returns zero matches |
| Performance parity | tensor-engine backend within 2x of ndarray backend on matmul benchmark |

---

## 8. Project Structure

```
rustml/
├── rustml-core/              # Tensor + autograd (ndarray backend, std)
│   └── src/
│       ├── tensor.rs         # Tensor struct, ops, in-place variants
│       ├── autograd.rs       # GradientTape (std-based)
│       └── pool.rs           # TensorPool (new, FR-400)
├── rustml-nn/                # Layers, attention, activations
│   └── src/
│       ├── backend.rs        # TensorBackend type alias (new, FR-300)
│       └── ...               # All layers use backend::Tensor
├── rustml-train/             # Training infrastructure
│   └── src/
│       └── trainer.rs        # Memory-bounded training loop
└── rustml-edge/              # Edge deployment utilities (new)
    └── src/
        ├── fine_tune.rs      # Layer freezing, LoRA adapter
        ├── checkpoint.rs     # Differential checkpointing
        └── federated.rs      # Gradient extraction/application

machinelearning/
└── tensor-engine/            # Custom tensor backend (no_std candidate)
    └── src/
        ├── api/tensor/       # Tensor struct, TensorOps trait
        ├── core/tensor/      # Math ops, SIMD kernels
        ├── core/autograd/    # GradientTape (new, alloc-only)
        └── core/pool.rs      # Arena allocator (new, no_std)
```

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| **Tier 1** | Linux-based edge device with `std` available (Pi, Jetson, industrial PC) |
| **Tier 2** | Bare metal or RTOS device requiring `no_std + alloc` (Cortex-M, RISC-V) |
| **TensorBackend** | Compile-time abstraction over tensor implementation (ndarray or custom) |
| **TensorPool** | Pre-allocated buffer recycling system with configurable memory ceiling |
| **Fine-tuning** | Training only a subset of model parameters on new data, with base frozen |
| **LoRA** | Low-Rank Adaptation — inject small trainable matrices into frozen layers |
| **Online learning** | Continuous weight updates from streaming data without epoch boundaries |
| **Federated learning** | Distributed training where data stays on-device; only gradients/deltas are shared |
| **Arena allocator** | Fixed-size memory pool that hands out buffers and reclaims them in bulk |
| **Differential checkpoint** | Save only parameters that changed since last checkpoint |
