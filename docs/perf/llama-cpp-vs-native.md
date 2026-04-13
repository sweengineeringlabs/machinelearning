# llama.cpp vs native-Rust backend — initial benchmark

First apples-to-apples comparison of the two `ModelBackendLoader`
impls behind `swellmd`. Run locally, single client, greedy decoding.
Not yet compared to Ollama's p50 directly — that's the next step.

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

| Backend      | p50 (ms) | p90 (ms) | p95 (ms) | req/sec | Wall (s) |
|--------------|---------:|---------:|---------:|--------:|---------:|
| `llama_cpp`  |   732.67 |   897.53 |   916.99 |    1.32 |     7.60 |
| `native_rust`|  5304.32 |  5406.72 |  5419.01 |    0.19 |    53.00 |

**Speedup: ~7.2× at p50, ~5.9× at p95.**

## Interpretation

The native-Rust backend serves the same prompts correctly — output
is coherent gemma3 text — but at roughly 1/7 the throughput. The gap
is almost entirely in per-token decode latency; prompt handling is
similar.

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

## Caveats

- **Different quantizations.** Q4_K_M vs F16 is not apples-to-apples
  at the weight level. The native-Rust path could be pushed toward
  Q8_0 or Q4_0 in its runtime quantizer to narrow this factor.
  Full K-quant (Q4_K_M) support in the native path is tracked as
  P7.5 in BACKLOG.md.
- **Single-client workload.** p50 numbers say nothing about
  contention. Multi-client tail latency is P7.B's concurrency
  question — unresolved until the context-pool decision from
  audit #3 is implemented.
- **Small sample.** 10 requests gives coarse-grained tail
  estimates. The p99/p99.9 columns in the table are pinned to the
  single max sample and shouldn't be read as real tail metrics.
- **Not yet compared to Ollama.** Ollama runs the same GGUF with
  the same llama.cpp vendored, so its p50 should be ~matching the
  `llama_cpp` row here. Running Ollama side-by-side is the next
  step to validate there's no daemon-level overhead we've added.

## Session date

2026-04-13 — corresponds to commits `db923b1` (llama-cpp-2 bindings),
`41710b2` (test fixture), `3e8dabb` (backend SPI).
