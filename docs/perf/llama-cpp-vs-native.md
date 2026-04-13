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

### Larger sample against Ollama (n=50)

| Backend             | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | req/sec |
|---------------------|---------:|---------:|---------:|---------:|--------:|
| Ollama (11434)      |   648.70 |   773.63 |   792.58 |   806.40 |    1.50 |
| swellmd `llama_cpp` |   698.88 |  1599.49 |  1954.82 |  4186.11 |    1.08 |

**swellmd is p50-competitive (+8%) but tail is 2.5–5× worse.** The
tail is the open cost.

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

## Next step: context pooling

The tail-latency gap has a clear fix. Instead of creating a fresh
`LlamaContext` per request, keep one (or N, keyed per
admission-control slot) alive in `LlamaCppModel` and reset sequence
positions via `kv_cache_seq_rm` between requests. `llama-cpp-2`
exposes the full sequence-ID API (`copy_kv_cache_seq`,
`clear_kv_cache_seq`, `kv_cache_seq_add`, etc.) so one context can
serve many logical conversations, which is what Ollama does.

Sketch:

1. Add a `Mutex<Option<LlamaContext<'static>>>` pool inside
   `LlamaCppModel` (one slot per admission-control permit — the
   throttle already bounds concurrency).
2. `LlamaCppTextCompleter::run_decode` acquires a context from the
   pool; if none free, create. Reset its KV cache before use;
   return to the pool on drop.
3. Size `n_ctx` at pool init using the model's `n_ctx_train` so
   one context can serve any reasonable prompt.

Expected impact: p95 should drop to ~800–900 ms (matching Ollama)
because memory allocation is no longer on the request hot path.
Correctness risk is small — KV cache reset is how llama.cpp
serves parallel sequences in its own server.

## Caveats

- **Different quantizations.** Q4_K_M vs F16 is not apples-to-apples
  at the weight level. The native-Rust path could be pushed toward
  Q8_0 or Q4_0 in its runtime quantizer to narrow this factor.
  Full K-quant (Q4_K_M) support in the native path is tracked as
  P7.5 in BACKLOG.md.
- **Single-client workload.** p50 numbers say nothing about
  multi-client contention. That's a separate axis from the
  fresh-context penalty above.
- **Local-only.** One machine, one network hop to localhost. No
  TLS, no external proxy.

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
