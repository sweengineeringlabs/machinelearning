# Glossary

Definitions for terms used across this codebase that benefit from clear origin and intent. Each entry is: **what it is**, **where the term comes from**, and **how we use it here**.

Keep this focused. Add terms when someone has to ask what one means, not preemptively.

## Semaphore

**What it is**: A synchronization primitive that maintains a counter of available permits. Operations: *acquire* (take a permit, blocks or fails if counter is zero) and *release* (return a permit, counter goes up). Classic mental model: a nightclub with N spots and a bouncer.

**Origin**: The name comes from railway signaling — a semaphore is a pivoting-arm signal used to indicate whether a section of track is clear. Edsger Dijkstra introduced it as a computer science concept in 1965 ("Cooperating Sequential Processes") to coordinate access to shared resources between concurrent processes. The two operations were originally named *P* (Dutch *passeren*, "to pass") and *V* (Dutch *vrijgeven*, "to release"). Modern APIs call them *acquire* and *release*.

**In this codebase**: `core/throttle.rs` uses `tokio::sync::Semaphore` (an async-aware implementation) as the backing store for `SemaphoreThrottle`. Capacity is configured at `[throttle.semaphore].max_concurrent` in `llminference/main/config/application.toml`. Request handlers call `try_acquire` (fail-fast variant) before `spawn_blocking`; failure returns HTTP 503 instead of queueing. See `llminference/main/features/inference/systemd/docs/3-design/architecture.md` for the full admission-control design.

Separately, `llmc load`'s open-loop mode uses a different semaphore to cap
in-flight requests during rate-targeted load generation — see below on
coordinated omission.

## Coordinated omission

**What it is**: A systematic tail-latency under-reporting bias in closed-loop load tests. If a server stalls for 5 seconds at a target rate of 100 req/s, the client — politely waiting for each request to return before sending the next — *omits* the 500 requests it should have sent during the stall. Only the requests that actually happened get measured, so the recorded tail hides the stall entirely.

**Origin**: Identified and named by Gil Tene (Azul Systems, ~2013) while benchmarking JVM garbage-collection pauses. The fix has since been adopted by `wrk2`, `tsung`, and most serious latency benchmarks. See Tene's talk *"How NOT to Measure Latency"* for the canonical treatment.

**In this codebase**: `llmc load --rate R` runs in open-loop mode: a scheduler dispatches one request every `1/R` seconds regardless of whether previous ones have finished. Each request's latency is measured from its **scheduled time**, not its actual send time. When the server stalls, requests queue up on the concurrency semaphore and their measured latency includes the wait, so percentiles reflect reality. Without `--rate`, `llmc load` runs closed-loop (workers fire as fast as possible) — suitable for "how fast can the server go?" questions but not CO-correct for SLO measurements.

## Permit

**What it is**: A handle representing the right to use a limited resource, issued by a semaphore or similar primitive. Holding the permit means you have a slot; dropping it releases the slot back.

**Origin**: Common English usage — a document that permits (allows) something. The programming term follows the semaphore metaphor: the bouncer *permits* you to enter.

**In this codebase**: `api/throttle.rs` defines a `Permit` struct whose `Drop` releases the underlying slot. The type is deliberately opaque — it wraps `Box<dyn Send + Sync>` so the trait doesn't leak tokio types to callers. Permits move into `spawn_blocking` closures so the slot auto-releases when work completes, even on panic.

## RAII

**What it is**: *Resource Acquisition Is Initialization.* A pattern where a resource's lifetime is tied to an object's lifetime: the object acquires the resource in its constructor and releases it in its destructor. Rust's `Drop` trait is RAII for resource cleanup.

**Origin**: Coined by Bjarne Stroustrup for C++ in the late 1980s. The name is famously awkward because the defining feature is actually the *release* on destruction, not the acquisition on construction — but the name stuck. Stroustrup has said he would rename it "Scope-Based Resource Management" if he could.

**In this codebase**: `Permit` uses RAII — the slot releases on drop, no explicit `release()` call needed. `Arc<Mmap>` uses RAII — the file unmaps when the last reference drops. `tokio::task::JoinHandle` does not use RAII by default, which is why we must be careful that spawned work completes before state it depends on is dropped.

## Throttle

**What it is**: A mechanism that limits the rate or concurrency of operations. In admission control, a throttle decides whether a new request is accepted right now or rejected.

**Origin**: Early 19th century English, originally the lever that controls steam flow to an engine — metaphorically extended to any flow-limiting device. In software, the term usually means either *rate limiting* (N ops per second) or *concurrency limiting* (N in-flight at once). We use it in the concurrency sense.

**In this codebase**: `api/throttle.rs` defines the `Throttle` trait with `try_acquire() → Option<Permit>`, `capacity()`, and `available()`. Implementations decide the backing mechanism. Currently only `SemaphoreThrottle` exists; future options could include token-bucket, distributed throttles, or `NoopThrottle` for tests.

## Memory-mapped file (mmap)

**What it is**: A mechanism that maps a file directly into a process's virtual address space. Reads of the mapped region are served by the OS pager, which reads file contents on demand. No explicit `read()` syscalls, no bulk heap allocation for the file.

**Origin**: Introduced in Multics in the 1960s, then standardized in BSD Unix (`mmap(2)`) in the 1980s. The key insight is that the OS already has machinery for paging memory — exposing that to user space lets programs treat files as if they were already in memory, with the kernel handling the physical-read side lazily.

**In this codebase**: `hub::load_safetensors` uses `memmap2::Mmap` to avoid allocating a 32 MB header buffer (plus hundreds of MB of weights) into the heap. `Tensor::from_mmap` holds an `Arc<Mmap>` so the tensor's data pointer remains valid until all tensors derived from that mapping are dropped. Critical caveat: if an external process truncates or rewrites the mapped file while the mapping is live, reads can return garbage or SIGBUS. We rely on the fact that model weights are read-only and not rewritten in place.

## KV cache

**What it is**: A cache of computed key and value projections from prior tokens in attention layers. Each new token only attends to itself plus the cached keys/values of previous tokens, so attention cost is O(seq_len) per new token instead of O(seq_len²) for the whole sequence.

**Origin**: Standard optimization in autoregressive transformer inference, popularized by the first generations of GPT serving infrastructure. The math is identical with or without the cache — caching trades memory for compute.

**In this codebase**: Allocated per request inside the inference path. Size for gemma-3-1b-it worst case: `2 × seq_len × num_kv_heads × head_dim × 4 bytes` per layer × 26 layers ≈ 416 MB per request at max context. This is why unbounded concurrent requests OOM'd before the `Throttle` was added — each request wanted its own cache.

## Quantization

**What it is**: Representing floating-point weights or activations in lower precision (8-bit, 4-bit) to save memory and enable faster integer arithmetic. Lossy, but often negligibly so for inference.

**Origin**: Signal processing term (analog → digital conversion), adapted to ML in the 2010s as models grew beyond single-GPU memory. The name is a bit misleading in ML context — we are not quantizing a continuous signal, we are re-encoding already-discrete f32 values at lower precision.

**In this codebase**: `rustml-quantizer` applies per-layer-type quantization configured in `quantization.toml`. Currently Q8_0 (8-bit with f16 scale per 32-element block, GGUF format) for most layers. See `docs/5-testing/report/perf-2026-04-12-quantization-speedup.md` for the measured impact.

## SafeTensors

**What it is**: A binary format for storing tensors. Header is a JSON blob at the file start describing shapes, dtypes, and byte offsets; tensor data follows contiguously. Designed to be safe to `mmap` and memory-safe to parse (no arbitrary code execution risk, unlike pickle).

**Origin**: Created by Hugging Face in 2022 as a safer alternative to PyTorch's `pickle`-based `.pt` / `.bin` files. The format specification lives at github.com/huggingface/safetensors.

**In this codebase**: Primary on-disk format for SafeTensors-native models downloaded from Hugging Face Hub. Loaded via the `safetensors` crate plus `memmap2` for zero-copy reads. See `hub::load_safetensors`.

## GGUF

**What it is**: *GPT-Generated Unified Format.* A binary format for storing quantized LLM weights along with tokenizer, model metadata, and chat templates in a single file. Supports many quantization schemes (Q8_0, Q4_0, Q4_K, Q5_K_M, etc.).

**Origin**: Created by Georgi Gerganov for `llama.cpp` in 2023, replacing the earlier GGML format. Designed specifically for quantized inference on consumer hardware.

**In this codebase**: Primary format for GGUF-native models. See `rustml-gguf::GGUFFile`. Zero-copy mmap loading via `GGUFFile::load_tensors_mmap`.

## SIMD

**What it is**: *Single Instruction, Multiple Data.* CPU instructions that operate on vectors of values in one cycle — e.g., AVX2's `_mm256_mul_ps` multiplies 8 f32 values simultaneously.

**Origin**: Classification introduced by Michael Flynn in 1966 ("Flynn's taxonomy" of parallel architectures: SISD, SIMD, MISD, MIMD). First commodity implementations were MMX (Intel, 1996) and SSE (Intel, 1999); AVX (2008) and AVX-512 (2013) widened the registers.

**In this codebase**: AVX2 is the baseline target on x86_64. Runtime detection at startup logs `[runtime] SIMD: AVX2`. Used in quantized dot products, softmax, RoPE, and other hot kernels. Non-SIMD scalar fallbacks exist for correctness.

## Tensor

**What it is**: A multi-dimensional array with a shape, a dtype, and contiguous (or strided) data. In ML the term is usually informal — we mean "n-dimensional array," not the full mathematical object.

**Origin**: The mathematical concept comes from late-19th-century differential geometry and physics (Ricci, Levi-Civita), where a tensor is a multi-linear map between vector spaces. ML borrowed the name for the data structure, dropping most of the math. Purists object; the usage is established.

**In this codebase**: `swe_ml_tensor::Tensor` is the core type — shape + dtype + `Arc<Storage>`, where storage is either heap-allocated bytes or an mmap-backed view. Supports F32, F16, BF16, Q8_0, Q4_0, Q4_1 dtypes. See `main/features/tensor/`.

## Rayon

**What it is**: A Rust data-parallelism library. Provides parallel iterators (`par_iter()`) that distribute work across a thread pool, and join-style parallelism (`rayon::join`).

**Origin**: Created by Niko Matsakis (Mozilla / Rust core team), released 2015. The name is a play on *Cilk*, an earlier work-stealing parallelism library — *rayon* is French for "ray" (as in a ray of sunshine, and as a callback to *Cilk* which is a kind of cloth / fiber). The work-stealing scheduler is essentially a Rust port of Cilk's design.

**In this codebase**: Used inside tensor operations (matmul, softmax, quantized dot products) for fork-join parallelism over batches, rows, or tiles. Thread-pool size configured via `rustml-thread-config`. Logged at daemon startup as `[runtime] Rayon threads: N`.
