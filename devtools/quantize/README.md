# rustml-quantize

Offline CLI that converts a HuggingFace SafeTensors model into a quantized GGUF file.

**This is a build-time tool, not a runtime component.** Nothing in the inference stack imports this crate. Operators run it once per model to produce an artifact; the daemon loads that artifact at startup.

## When to use it

Run this tool when you want to:

- **Shrink model size on disk** — Q8_0 is ~4× smaller than F32, Q4_0 is ~8× smaller. For a 1 B-parameter model: F32 ≈ 5 GB, Q8_0 ≈ 1.3 GB, Q4_0 ≈ 700 MB.
- **Skip runtime quantization** — the daemon's `quantizer` library does the same math at model load (~10 s for Gemma-3-1B). Pre-quantizing moves that cost from every boot to one-off.
- **Reduce peak load memory** — runtime quantization briefly holds F32 + quantized in memory. Offline path never materializes F32.
- **Produce a portable artifact** — one GGUF file you can hash, version, and ship. Consumable by any GGUF-compatible runtime (this daemon, llama.cpp, Ollama, etc.), not just ours.

You do **not** need this tool for correctness — the daemon can run directly from SafeTensors with runtime quantization. This is an optimization.

## Install

Built as part of the workspace:

```bash
cargo build --release -p rustml-quantize
# binary at target/release/rustml-quantize
```

## Usage

```bash
rustml-quantize --model <HF_REPO> --target <q8_0|q4_0|q4_1> --output <PATH.gguf> [flags]
```

### Example

```bash
# Quantize Gemma 4 E2B to Q4_0
rustml-quantize \
  --model google/gemma-4-e2b-it \
  --target q4_0 \
  --output ./models/gemma-4-e2b-q4_0.gguf
```

Then start the daemon against the GGUF file:

```bash
swellmd --gguf-path ./models/gemma-4-e2b-q4_0.gguf
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--model <HF_REPO>` | required | HuggingFace repo ID, e.g. `google/gemma-3-1b-it`. Downloaded via `rustml-hub` if not cached. |
| `--target <FORMAT>` | `q8_0` | Quantization target: `q8_0`, `q4_0`, `q4_1`. |
| `--output <PATH>` | required | Path to write the quantized GGUF file. |
| `--preserve-output` | off | Keep the final `lm_head` / output projection in F32. Reduces output perplexity at the cost of a larger file. |
| `--min-dim <N>` | `0` | Skip tensors with any dimension smaller than `N`. Small tensors quantize poorly and the size saving is negligible. |
| `--metrics` | off | Print per-tensor quantization error (MSE, max absolute deviation). Useful for comparing targets. |

## Quantization targets

| Target | Bits per weight | Typical size vs F32 | Use when |
|---|---|---|---|
| `q8_0` | 8 (block of 32 + f16 scale) | ~0.26× | Near-lossless baseline; quality is indistinguishable from F16 on most tasks. |
| `q4_0` | 4 (block of 32 + f16 scale) | ~0.14× | Size matters. Some quality loss, especially on small models. |
| `q4_1` | 4 (block of 32 + f16 scale + f16 min) | ~0.15× | Slightly more accurate than `q4_0`, slightly larger. |

## What it produces

A single GGUF file containing:

- Quantized tensor weights in the target format
- Original metadata (architecture, hyperparameters, hidden sizes, etc.)
- Tokenizer data, so the daemon can tokenize without a separate `tokenizer.json`
- Chat template, if present in the source model

The file is self-contained — once produced, you can delete the original SafeTensors download if you only need the quantized version.

## Failure modes

- **Network**: downloads from HuggingFace Hub on first run. `HF_TOKEN` required for gated models.
- **Disk**: writes the GGUF file in one pass. Needs free space equal to the output size plus a working buffer.
- **Unsupported architectures**: quantization uses GGUF tensor naming conventions; models with non-standard naming may fail with `unknown tensor` errors. Check `rustml-gguf` for supported architectures.

## Relationship to `rustml-quantizer`

The names sound similar. The crates do different things:

| Crate | Where it runs | What it takes | What it produces |
|---|---|---|---|
| `rustml-quantize` (this tool) | offline, once, by a human | HF SafeTensors model | GGUF file on disk |
| `rustml-quantizer` (library, in inference stack) | at daemon startup, every boot | F32/F16 weights in memory | Quantized weights in memory |

Both implement the same quantization math. You can use either path — or both — to the same end.

## Project layout

```
devtools/quantize/
├── Cargo.toml
├── README.md            (this file)
├── main/src/
│   ├── lib.rs           — re-exports the engine API for tests
│   ├── main.rs          — CLI entry point
│   ├── api/             — public types (QuantizeConfig, QuantTarget, QuantizeReport)
│   ├── core/            — engine implementation
│   └── saf/             — facade re-exports
├── examples/            — runnable demos
└── docs/                — design notes and detailed guides
```
