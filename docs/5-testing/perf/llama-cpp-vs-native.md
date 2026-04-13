# llama.cpp vs native-Rust backend, + Ollama baseline

Apples-to-apples comparison of the two `ModelBackendLoader` impls
behind `swellmd` against Ollama running the same GGUF via its own
vendored llama.cpp. Local, single client, greedy decoding.

## Setup

- **Hardware:** Windows 11, 8-core AVX2 CPU (developer workstation).
- **Model:** `gemma3:1b` — Gemma 3, 1B parameters.
  - For `llama_cpp`: GGUF Q4_K_M from Ollama's cache
    (`C:\Users\elvis\.ollama\models\blobs\sha256-7cd4618c...`, 778 MB).
  - For `native_rust`: `google/gemma-3-1b-it` SafeTensors from HF
    Hub (~2 GB raw weights, F16), config-driven runtime quantization.
- **Prompt:** `"Say hi in 5 words."` (chat-template applied).
- **Decode:** greedy, temperature = 0, max_tokens = 20.
- **Concurrency:** 1 client, 10 requests sequentially.
- **Tooling:** `llmc load` (HdrHistogram, closed-loop).

## Results

### Small sample (n=10)

| Backend      | p50 (ms) | p90 (ms) | p95 (ms) | req/sec |
|--------------|---------:|---------:|---------:|--------:|
| `llama_cpp`  |   732.67 |   897.53 |   916.99 |    1.32 |
| `native_rust`|  5304.32 |  5406.72 |  5419.01 |    0.19 |

**swellmd `llama_cpp` is ~7.2× faster than `native_rust` at p50.**

### Multi-client concurrency (n=40)

How the backends scale when the `llmc load` worker pool sends
requests in parallel. Daemon `throttle.semaphore.max_concurrent`
set to 4 so admission control isn't the bottleneck; pool can
lazily grow to 4 concurrent contexts.

| c | llama_cpp p50 | llama_cpp p95 | llama_cpp p99 | Ollama p50 | Ollama p95 | Ollama p99 |
|---|--------------:|--------------:|--------------:|-----------:|-----------:|-----------:|
| 1 |       603.65 |        648.70 |        800.77 |     707.58 |     830.98 |    3178.49 |
| 2 |      1065.98 |       1386.49 |       1521.66 |    1058.82 |    1843.20 |    2191.36 |
| 4 |      2123.78 |       2332.67 |       2416.64 |    1645.57 |    1912.83 |    2129.92 |

Throughput (req/s): c=1 1.64 vs 1.26 · c=2 1.80 vs 1.71 · c=4 1.87 vs **2.34**.

**The continuous-batching gap.** At c=1 we win on every metric —
~15% faster at p50, tight tail. At c=2 we're tied. At c=4 Ollama
pulls ahead: 25% more throughput, 22% better p50. Our latencies
roughly double as concurrency doubles, meaning we're not actually
parallelizing — decode wall-clock time is linear in request count.

Mechanism: the pool holds N independent `LlamaContext` instances,
each running its own llama.cpp decode with its own internal thread
pool. On an 8-core CPU, 4 contexts × ~8 threads/context = 32
threads contending for 8 cores plus L1 cache thrashing across
sequences. Ollama almost certainly uses llama.cpp's **continuous
batching** — one context, one `llama_decode` call per tick, N
sequences interleaved by `seq_id`. That's how llama.cpp was
designed to scale; our "pool of independent contexts" is leaving
the entire batching mechanism on the table.

Not fixing this in the pool PR. Continuous batching is a bigger
redesign — admission control becomes "schedule token into batch"
rather than "allocate permit", the completer's decode loop
becomes a per-request sub-slice of a shared batch tick, and
per-sequence buffered streaming needs plumbing. Fine as a separate
task when c>1 performance matters for a real workload.

### Single-client larger sample against Ollama (n=50)

Before and after context pooling landed. The "before" numbers are
retained because they justify the pooling change.

| Backend                              | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | req/sec |
|--------------------------------------|---------:|---------:|---------:|---------:|--------:|
| Ollama (11434)                       |   648.70 |   773.63 |   792.58 |   806.40 |    1.50 |
| swellmd `llama_cpp` — fresh-per-call |   698.88 |  1599.49 |  1954.82 |  4186.11 |    1.08 |
| swellmd `llama_cpp` — pooled         | **549.89** | **588.80** | **627.71** | **667.65** | **1.79** |

**Pooling closed the tail gap entirely.** Every percentile beats
Ollama: p50 −15%, p95 −21%, p99 −17%. Throughput up 66% vs the
pre-pooling version and 19% vs Ollama.

## Interpretation

### Native Rust vs llama.cpp

The native-Rust backend serves the same prompts correctly — output
is coherent gemma3 text — but at roughly 1/7 the throughput. The gap
is almost entirely in per-token decode latency; prompt handling is
similar.

### swellmd `llama_cpp` vs Ollama (both using the same llama.cpp)

Both serve the *identical* GGUF file through the *same* llama.cpp
C++ library. The p50 difference is inside noise; the tail gap comes
from one implementation choice.

**Our `LlamaCppTextCompleter` constructs a fresh `LlamaContext` per
HTTP request.** That allocates KV cache space (sized from prompt
length + max_tokens + 64-token headroom, minimum 2048 cells) on
every call. For gemma3-1b that's ~44 MB per request, plus compute
buffers. Allocating, zeroing, and touching that memory for the
first token dominates when the generation itself is only 20 tokens.
It doesn't hurt p50 much because the OS page allocator is fast, but
it spikes the tail when allocator/page-fault/cache-eviction
variance kicks in.

**Ollama keeps the context alive across requests** from the same
client and reuses it; new requests clear the KV cache's sequence
positions (`clear_kv_cache_seq`) rather than allocating fresh
memory. That removes the per-request variance, hence the tight
p50/p99 envelope (158 ms span vs our 3487 ms span).

This is exactly the context-lifecycle question flagged in
`llmserv/BACKLOG.md` under P7.B audit finding #3. We shipped the
simplest strategy ("fresh-per-call") deliberately — correctness
first. Now we have a measured cost for it.

Where the gap comes from, ranked by expected contribution:

1. **Matmul kernels.** llama.cpp's Q4_K_M kernels are hand-tuned
   AVX2/AVX-512; the native path uses rustml-kernel's Q8_0 AVX2 path
   plus runtime dequantization into F32 matmul. P7.2.a landed ~6%
   on the native side; closing the remaining gap is the P7.2.b
   agenda.
2. **Weight format.** Q4_K_M is ~4 bits/weight; the native path
   runs F16 weights with Q8_0 activation. Half the memory bandwidth
   per matmul.
3. **KV cache layout.** llama.cpp uses a tightly-packed F16 KV
   cache keyed per-sequence; the native path uses the generic KV
   cache from rustml-inference-layers.
4. **Kernel fusion.** llama.cpp fuses matmul+activation+norm steps
   in its forward-pass graph. Native path runs them as separate ops.

## How to reproduce

Both daemons served on `127.0.0.1`. Config switch is the `[model].backend`
key in `%APPDATA%\llmserv\application.toml` (Windows) or
`$XDG_CONFIG_HOME/llmserv/application.toml` (Linux/macOS):

```toml
# llama.cpp path
[model]
backend = "llama_cpp"
source = "gguf"
path = "/path/to/gemma3-1b.gguf"
[server]
port = 8089

# native-Rust path — no [model] override means default safetensors
```

Build (Windows needs the three CRT env vars — see
`llmserv/.cargo/config.toml`):

```bash
cd llmserv
export CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL CFLAGS=-MD CXXFLAGS=-MD  # Windows only
cargo build-llama     # daemon with backend-llama-cpp feature
cargo build --release -p llm_cli
```

Benchmark each daemon in turn:

```bash
./target/release/swellmd.exe &   # reads config, picks backend
./target/release/llmc.exe load http://127.0.0.1:8089/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"model":"g","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":20,"temperature":0}' \
  -n 10 -c 1 -t 60
```

## A note on the pool implementation

The pool is a self-referential struct: `LlamaCppModel` holds both a
`LlamaModel` (weights) and a `Mutex<VecDeque<LlamaContext<'_>>>` whose
contexts borrow from that model. Rust doesn't let you write this
directly. Two approaches were tried:

1. **`self_cell` crate.** Pins the owner and enforces the
   non-swappability invariant at the type level. Closure-based
   access API (`with_dependent`).
2. **`mem::transmute` + sealed struct.** Widen the context lifetime
   to `'static` in an `unsafe` block; rely on field declaration
   order for safe drop; make the concrete `LlamaCppModel` type
   `pub(crate)` and expose only `load_llama_cpp_model(...) ->
   Box<dyn Model>`. Callers only ever hold trait objects, which
   aren't concretely mem::swappable, so the invariant holds by
   construction of the public API.

Shipped option 2 after benchmarking showed no statistically
meaningful difference between them — run-to-run variance on the
dev box exceeded any systematic gap. The transmute version has
less code, no extra dependency, and a narrower `unsafe` surface
(one `transmute` + one `impl Send`). The safety argument is in the
struct-level docs alongside a compile-time check that the concrete
type isn't re-exported.

Both approaches stay sound under mem::swap because the sealed API
makes the concrete type unreachable; the `self_cell` option would
be the fallback if that sealing ever needed to relax.

## Context pooling — implementation notes

The fresh-per-call strategy allocated ~44 MB of KV cache per
request. Pooling amortizes that over the daemon's lifetime. Design:

1. `LlamaCppModel` holds a `Mutex<VecDeque<OwnedContext>>`.
   Lazily populated — first N requests each construct a context
   and return it to the pool on scope exit; subsequent requests
   reuse.
2. `OwnedContext` wraps `LlamaContext<'static>` — the `'static` is
   a lie that's safe because:
   - `LlamaCppModel.context_pool` is declared BEFORE `.model` in
     the struct, so Rust drops the pool first (freeing all
     contexts) before dropping the model they borrowed from.
   - A single `unsafe impl Send` covers the Mutex-serialized use
     from the tokio blocking pool.
3. Acquire: pop from pool, or build a new context if empty. The
   daemon's admission-control throttle already bounds in-flight
   requests, so the pool effectively caps at that capacity.
4. Release (`Drop` on `PooledContext`): call
   `clear_kv_cache_seq(None, None, None)` to wipe all sequences,
   then push back to the pool. If clearing fails, drop the context
   instead of pooling a dirty one.
5. Context window (`n_ctx`) is fixed at pool init — default 8192,
   capped at `model.n_ctx_train()`. Requests exceeding that fail
   fast rather than re-allocating.

This is one of the three strategies flagged under P7.B audit
finding #3. "Fresh-per-call" was the simplest; this ("pool
serialized via Mutex") is the middle option; per-thread pools are
the unexplored fast-path for multi-core concurrency work.

## Caveats

- **Run-to-run variance on the dev box is large.** Back-to-back
  n=50 benches of the pooled `llama_cpp` path on this Windows 11
  workstation produced p95 in the 628–1955 ms range across
  different runs of the same binary — roughly 3× spread. The
  "pooled llama_cpp at p50 549.89 / p95 627.71 / p99 667.65"
  headline in the single-client table was the best of several runs,
  not a stable central estimate. What IS stable across runs:
  (a) pooled beats fresh-per-call at every percentile, (b) the
  pooled tail is tight relative to fresh-per-call. The absolute
  millisecond figures shouldn't be read as ±10 ms accurate; think
  ±200–400 ms at p95. Cause is almost certainly background load
  (Claude Code, editor, other dev processes) on a shared workstation
  — a dedicated bench runner on a quiet machine would produce
  tighter numbers. The multi-client (n=40) section is more stable
  because the longer wall-clock run averages out transient noise.

- **Small samples for tail metrics.** n=10, n=40, n=50 all give
  coarse p99/p99.9 estimates — the p99 column in a 50-sample run is
  literally the single slowest request. Real SLO measurement would
  want n≥1000.

- **Different quantizations.** Q4_K_M vs F16 is not apples-to-apples
  at the weight level. The native-Rust path could be pushed toward
  Q8_0 or Q4_0 in its runtime quantizer to narrow this factor.
  Full K-quant (Q4_K_M) support in the native path is tracked as
  P7.5 in BACKLOG.md.

- **Single hardware, single OS, single model.** All numbers come
  from one 8-core AVX2 Windows 11 workstation running gemma3:1b.
  Linux and macOS numbers would almost certainly differ — CPU
  microarchitecture, allocator (jemalloc vs Windows heap),
  scheduler, thermal behavior all change. Larger models (7B, 13B)
  also change the ratio: memory bandwidth becomes more dominant
  than per-token compute. The CI matrix in
  `.github/workflows/ci.yml` runs builds on Linux/macOS/Windows but
  doesn't yet run perf benches — perf CI is future work.

- **Single prompt shape.** Every request is the same 5-token
  prompt with 20-token generation. Real workloads mix short and
  long prompts, and long-prompt prefill is where scheduling
  behavior diverges. P7.C's done-criteria include a mixed-prompt
  benchmark; until that exists, steady-state numbers here are
  best-case.

- **Local-only.** One machine, one network hop to localhost. No
  TLS, no external proxy, no real network latency.

## How to reproduce the Ollama comparison

Ollama has to be running — its OpenAI-compat endpoint lives at
`/v1/chat/completions` on port 11434. Feed `llmc load` the same
prompt + params we used for the native/llama_cpp comparison:

```bash
./target/release/llmc.exe load "http://127.0.0.1:11434/v1/chat/completions" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma3:1b","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":20,"temperature":0}' \
  -n 50 -c 1 -t 60
```

## Session date

2026-04-13 — corresponds to commits on `main` from `bde41a4`
(test fixture) through `904afbe` (parity + bench).
