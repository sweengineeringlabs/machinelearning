# WeightScope — Rust Quantization

A toolkit for AI model weight quantization and quality auditing.

**Status:** Phase 1 (Int8 only). NF4, Int4, Q4_0, and Fp16 are listed in the
roadmap but **not implemented** — see [docs/4-development/code_review.md](docs/4-development/code_review.md)
for the audit history that surfaced earlier overstated claims.

---

## Pipeline Architecture (SEA-layered)

| Crate              | Stage             | Status                                                                                                          |
| :----------------- | :---------------- | :-------------------------------------------------------------------------------------------------------------- |
| `quant-api`        | Foundation        | ✅ `Quantizer` trait, `QuantFormat::Int8`, `QuantizedTensor`, `QuantError`                                       |
| `quant-io`         | Load / save       | ✅ `ModelIO::{load_safetensors, save_safetensors, load_gguf_tensor}`                                             |
| `quant-packer`     | Bit-packing       | ⚠️ Empty placeholder crate — reserved for 4-bit bit-packing when NF4/Q4_0/Int4 land                            |
| `quant-engine`     | Quantization math | ✅ `DefaultQuantService` — Int8 block-symmetric (per-block f32 scale)                                            |
| `quant-eval`       | Quality metrics   | ✅ `EvalService` returns `Metrics { snr_db, cosine, mse, max_abs_error }`                                        |
| `quant-cli`        | CLI               | ✅ `--format int8`, `--block-size`, `--verify`, `--protect-report`                                              |
| `sweetspot-finder` | Research          | ✅ Block-size sweep over Int8                                                                                    |

---

## Quick Start

### Quantize and verify
```bash
cargo run -p quant-cli -- \
    --input model.safetensors \
    --output model.int8.safetensors \
    --format int8 \
    --block-size 128 \
    --verify
```

`--verify` round-trips every tensor through the quantizer and aborts with a
non-zero exit code if any tensor's cosine similarity drops below `0.99`.
Average SNR is printed at the end.

### Sweep block sizes
```bash
cargo run -p sweetspot-finder -- \
    --input model.safetensors \
    --sweep "32,64,128,256"
```

### GGUF audit (early)
```bash
cargo run -p quant-cli -- \
    --input model.gguf \
    --reference model.safetensors
```

---

## Round-trip contract

`Quantizer::dequantize(quantize(t))` must produce a tensor of the same shape
and dtype as `t`, with cosine similarity above the format's documented
tolerance:

| Format | Cosine threshold | Test                                                                                              |
| :----- | :--------------- | :------------------------------------------------------------------------------------------------ |
| Int8   | ≥ 0.99           | `quant-engine::test_int8_quantize_dequantize_recovers_known_signal_with_high_cosine_similarity`   |

This contract is exercised by per-crate unit tests **and** an end-to-end
smoke test (`quant-cli/tests/cli_verify_e2e.rs`) so a regression in either
the engine or the CLI plumbing is caught.

---

## Honest limitations

- Saved Int8 safetensors files are write-only by this CLI today: the
  packed payload is stored alongside `{name}.scales` but no reverse loader
  exists yet.
- GGUF support is read-only and minimal — only single-tensor lookup.
- The legacy SEA scaffolding (`Service` trait, `DefaultService` per crate)
  generates dead-code warnings; it predates the real implementations and
  will be removed in a follow-up.

*Built with Rust and Candle.*
