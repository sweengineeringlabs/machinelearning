# Load Testing Report: swellmd Daemon Concurrency

> **Date**: 2026-04-12
> **Platform**: Windows 11, x86_64, AVX2, 8 rayon threads
> **Model**: google/gemma-3-1b-it (1.3B params, SafeTensors, Q8_0 quantized, mmap-loaded)
> **Binary**: `target/release/swellmd.exe --safetensors google/gemma-3-1b-it --port 8080`
> **Client**: `llmserv/main/features/daemon/scripts/load_test.sh N` (bash + curl, N parallel subshells)

## Summary

The daemon serves concurrent requests reliably up to at least N=16, with total throughput scaling up to ~core count before plateauing. At N=50, the process crashes with an OOM on a 32 MB allocation. Root cause: **no admission control** — every request kicks off `tokio::task::spawn_blocking` immediately, so a burst lets every request simultaneously allocate its own KV cache, logits buffers, and activations. The tokio blocking pool defaults to 512 threads, so there is nothing throttling the burst.

## Methodology

- Single server instance, fresh start per test round.
- Each request: `POST /v1/chat/completions` with `max_tokens=8`, `temperature=0.0`, one user message.
- All requests fired in parallel via backgrounded bash subshells; wall clock measured from first fork to last `wait`.
- Per-request latency from `curl` timing; success = HTTP 200.
- Server health verified before and after each run via `GET /health`.

## Results

| N concurrent | Success | Wall clock | Per-req min/avg/max | Per-req tok/s | Total tok/s | Server alive |
|---|---|---|---|---|---|---|
| 1  | 1/1   | 3.4s  | 3.4 / 3.4 / 3.4     | 2.4 | 2.4 | yes |
| 4  | 4/4   | 7.7s  | 7.3 / 7.5 / 7.6     | 0.5 | 2.1 | yes |
| 8  | 8/8   | 15.0s | 14.3 / 14.6 / 14.8  | 0.3 | 2.1 | yes |
| 16 | 16/16 | 35.3s | 32.7 / 34.0 / 34.9  | 0.1 | 1.8 | yes |
| 50 | 0/50  | 10.4s | —                    | —   | —   | **crashed** (OOM) |

Server log at crash:
```
[INFO swellmd::core::router] Generated 8 tokens in 3.38s (2.4 tok/s)
memory allocation of 33554432 bytes failed
```

Failed requests after crash returned HTTP 000 (connection refused), not 5xx — the process was dead.

## Analysis

### Concurrency scales up to core count, then plateaus

Total throughput at N=1..16 stays in the 1.8–2.4 tok/s band. Per-request latency grows roughly linearly because each request's rayon-parallelized matmul competes with every other request for the same 8 cores. Once concurrency matches core count (N≈8), additional concurrency only adds queue depth, not throughput.

This is expected for CPU inference: the generation loop is already multi-threaded via rayon inside each request. Stacking more requests on top oversubscribes the CPU.

### The 32 MB allocation failure

`33554432 = 32 × 1024 × 1024`. Inside the inference path, this is most consistent with a per-request logits or activation buffer — at vocab_size=262144 and F32 elements, one logits row is ~1 MB; 32 tokens of prefill logits = 32 MB. Under 50 concurrent requests, ~50 of these allocate nearly simultaneously, alongside per-request KV caches (~416 MB worst-case per request at max context). The allocator eventually fails on one 32 MB request and the process aborts.

The exact allocation site wasn't pinpointed — no backtrace was captured. A future test with `RUST_BACKTRACE=1` will locate it.

### No admission control

`llmserv/main/features/daemon/main/src/core/router.rs` calls `tokio::task::spawn_blocking` directly in both `handle_blocking` and `handle_streaming`. There is no semaphore, no queue, no per-request timeout governor. The tokio blocking pool has a 512-thread default ceiling, so 50 requests all get CPU immediately and compete for memory.

## Recommendations

1. **Add a concurrency limit** in `AppState` via `Arc<Semaphore>`. Acquire a permit before `spawn_blocking` in chat and embeddings handlers. A good default for CPU inference is `num_cpus / rayon_threads_per_request`, which for the current config (rayon uses all 8) means permits ≈ 1–2. Requests beyond the limit queue in the semaphore wait list and are served as permits free.
2. **Capture the exact allocation site** under load with `RUST_BACKTRACE=full`. Once identified, consider pooling the buffer (e.g. reuse one logits buffer per worker thread) so concurrent requests share rather than each allocating fresh.
3. **Return 503 on queue overflow** once a concurrency limit exists, so clients get a proper backpressure signal instead of a TCP RST after the process dies.

## Reproducing

```bash
# Start server
target/release/swellmd.exe --safetensors google/gemma-3-1b-it --port 8080

# Fire N concurrent requests
./llmserv/main/features/daemon/scripts/load_test.sh 16   # or any N

# N > ~20 on Windows/Cygwin may hit fork exhaustion — that's a client-side
# limit, not a server issue. For large N, use a Python or Rust client.
```

Output lands in `llmserv/main/features/daemon/scripts/out/` (gitignored): one `stat_$i.txt` and `resp_$i.json` per request.

## Fix: admission control via `Throttle` trait

Added a `Throttle` trait (`api/throttle.rs`) with a `SemaphoreThrottle` implementation (`core/throttle.rs`). `AppState` now holds `Box<dyn Throttle>`. Every compute handler (`chat_completions` blocking, `chat_completions` streaming, `embeddings`) calls `try_acquire()` before `spawn_blocking`. On `None`, the handler returns `DaemonError::AtCapacity` → HTTP 503 with body `{"error":{"message":"Server at capacity (max concurrent=N)","type":"at_capacity"}}`. The permit moves into the `spawn_blocking` closure and drops when the work finishes, freeing the slot.

New CLI flag: `--max-concurrent N` (default: 2).

### Results after fix

| N concurrent | `--max-concurrent` | 200 | 503 | Server alive |
|---|---|---|---|---|
| 20 | 2 | 2  | 18 | yes |
| 30 | 2 | 2  | 28 | yes |
| 30 | 8 | 8  | 22 | yes |

503 latency is sub-second (failed fast, no memory allocated). 200 requests complete as before. The process no longer crashes under burst load.

### Unit tests

`core::throttle::tests` (3 tests, all passing):
- `test_try_acquire_admits_up_to_capacity_then_rejects` — N permits granted, N+1 rejected
- `test_dropping_permit_releases_slot` — permit Drop returns the slot
- `test_new_clamps_zero_to_one` — capacity=0 is treated as 1 (can't have a dead server)

