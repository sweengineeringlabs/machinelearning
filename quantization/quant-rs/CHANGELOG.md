# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `quant-api`: real public surface — `Quantizer` trait, `QuantFormat::Int8`,
  `QuantizedTensor`, `QuantError` (with `From<candle_core::Error>`).
- `quant-engine`: `DefaultQuantService` implementing block-symmetric Int8
  quantization (per-block f32 scale, zero-block fallback, configurable
  `block_size`). Round-trip on a sin-wave fixture achieves cosine ≥ 0.99.
- `quant-eval`: `EvalService::calculate_metrics` returning
  `{ snr_db, cosine, mse, max_abs_error }` with f64 accumulators internally.
- `quant-io`: `ModelIO::{load_safetensors, save_safetensors,
  load_gguf_tensor}` with `tempfile`-backed regression tests.
- `quant-cli`: `--verify` flag — quantize → dequantize → assert
  `cosine ≥ 0.99` per tensor; aborts with the offending tensor names on
  failure. End-to-end smoke test under `tests/cli_verify_e2e.rs`.
- `docs/4-development/code_review.md`: ongoing audit of the project,
  including the historical record of the empty-scaffolding state.

### Fixed
- `quant-cli/Cargo.toml` had a duplicate `safetensors.workspace = true`
  declaration that prevented `cargo metadata` from loading any workspace
  manifest.
- `quant-cli` and `sweetspot-finder` referenced `QuantFormat::{Nf4, Int4,
  Fp16, Q4_0}` and `Metrics::kl_divergence` — symbols that did not exist
  anywhere in the tree. Both binaries now compile against the real surface
  and are restricted to the formats/metrics that have backing
  implementations.

### Removed
- Empty `pub fn quantize() {}` / `pub fn dequantize() {}` SAF facade stubs
  in `quant-api`, `quant-engine`, `quant-eval`, `quant-io`. The SAF layer
  now re-exports the actual public types.

### Honest scope notes
- NF4, Int4, Q4_0, and Fp16 are **not** implemented. The earlier
  `feat(quantization): add WeightScope Rust quantization pipeline` commit
  (`9a5bbd5`) overstated the project state — see
  `docs/4-development/code_review.md` for the audit. They will be added
  one at a time, each behind a passing round-trip test.
- The legacy generic `Service` / `DefaultService` per-crate scaffolding is
  intentionally left in place for now; it generates dead-code warnings
  but does not interfere with the real types.
