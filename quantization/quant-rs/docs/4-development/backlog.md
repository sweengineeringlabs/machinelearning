# quant-rs Backlog

Open work, ordered by rough dependency. Each item lists concrete
acceptance criteria so a future session (or agent) can close it without
re-scoping. History of completed work lives in `CHANGELOG.md` and the
status tables in `code_review.md`.

---

## Round-trip correctness — additional formats

Every new format must land together with a passing round-trip test that
would fail if the implementation is wrong. Latency alone is never an
acceptable basis for claiming "done" — see
`feedback_content_correctness_required`.

- [ ] **NF4 (Dettmers 2023) end-to-end** — Add `QuantFormat::Nf4` and an
  `Nf4Quantizer` implementing `Quantizer`. Per-block absmax scaling, 16
  fixed levels. Pack two 4-bit codes per byte via a new
  `quant-packer` helper (move the bit-packing out of the quantizer).
  Acceptance:
  - sin-wave round-trip: cosine ≥ 0.99 at `block_size = 64`
  - normal-distribution round-trip (10k samples): MSE ≤ 0.002
  - e2e CLI test: `--format nf4 --verify` passes
  - Files (expected): `quant-api/src/api/format.rs`,
    `quant-engine/src/core/quantizer/nf4.rs`,
    `quant-packer/src/core/pack_4bit.rs`,
    `quant-cli/src/main.rs`.

- [ ] **Q4_0 matching llama.cpp layout** — Add `QuantFormat::Q4_0`. One
  f16 scale per 32-element block, values stored as signed 4-bit offset
  by +8. Byte layout must be bit-for-bit identical to a GGUF Q4_0
  block so llama.cpp can load our output.
  Acceptance:
  - round-trip on sin wave: cosine ≥ 0.98 at `block_size = 32` (Q4_0 is
    lossier than Int8)
  - byte-layout check: write a known-input Q4_0 block, compare against a
    golden reference vector (hand-computed or cross-checked against
    `candle-core::quantized::k_quants` if exposed)
  - CLI `--format q4_0 --verify` passes
  - Files: same layout as NF4 entry

- [ ] **Int4 symmetric** — Simpler cousin of NF4: signed 4-bit with
  per-block f32 scale. Reuses the packer from NF4. Mostly interesting
  for size comparisons.
  Acceptance:
  - sin-wave round-trip cosine ≥ 0.97
  - CLI `--format int4 --verify` passes

- [ ] **Fp16 passthrough** — Not really quantization, but listed in the
  original CLI enum. Just down-casts F32 → F16. Ensures the matrix of
  supported dtypes stays explicit instead of buried in conversions.
  Acceptance:
  - round-trip cosine ≥ 0.9999 (lossy only to f16 precision)
  - shape + dtype identical after round-trip

---

## IO & persistence

- [ ] **Reverse loader for Int8 quantized safetensors** — Today the CLI
  writes `{name}` (U8 payload) and `{name}.scales` (F32 scales) but
  discards the original shape and format. Without these, a saved file
  cannot be dequantized back to F32 outside the session that produced
  it. Design call needed between:
  - (a) sibling tensors: `{name}.shape` (I64), `{name}.format` (U8 enum)
  - (b) safetensors metadata map (string → string), parsed by ModelIO
  - (c) a new `QuantizedSafeTensors` wrapper format
  Acceptance:
  - `ModelIO::load_quantized_safetensors(path)` returns
    `HashMap<String, QuantizedTensor>` with format + shape intact
  - round-trip through disk: `save → load → dequantize` gives cosine
    ≥ 0.99 against the original F32
  - interop: a file produced by our save loads back without referencing
    the producing process

- [ ] **GGUF multi-tensor enumeration** — `ModelIO::load_gguf_tensor`
  currently takes a name; callers cannot discover what's in a file.
  Add `ModelIO::list_gguf_tensors(path) -> Vec<String>` so the CLI's
  GGUF audit mode can iterate without hardcoded names.

---

## Tooling & CLI

- [ ] **Negative-path `--verify` with a mock quantizer** —
  `test_cli_verify_aborts_when_threshold_is_unsatisfiable` proves the
  abort path via `--verify-threshold 1.01`. That's honest but trivial.
  A stronger test would inject a deliberately broken quantizer (e.g.
  one that zeros out every other element) via an env-var-gated DI
  point, then assert the CLI reports the correct tensor. Requires a
  small SPI refactor in `quant-engine` to make the quantizer
  constructor swappable at runtime.

- [ ] **`ModelIO::save_quantized_safetensors` convenience** — Today
  `quant-cli` hand-assembles a `HashMap<String, (Dtype, Vec, Vec)>`.
  Once the reverse loader lands, this belongs behind a named method
  that also writes whatever metadata (a) / (b) / (c) was chosen.

---

## Cleanup & hygiene

- [x] **Delete legacy `Service` / `DefaultService` / `Config` / `Error`
  scaffolding and `gateway/` directories** — done 2026-04-15. Removed
  across all seven crates. Zero build warnings from this tree now.

- [ ] **`quant-packer` is empty** — kept as a placeholder for future
  4-bit bit-packing. If NF4/Q4_0/Int4 all land inside `quant-engine`
  instead, delete the crate entirely to stop advertising a module that
  exists only in the Cargo.toml members list.
