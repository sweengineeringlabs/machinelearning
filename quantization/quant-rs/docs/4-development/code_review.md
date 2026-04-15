# quant-rs Code Review

**Date:** 2026-04-15
**Reviewed commit:** `9a5bbd5 feat(quantization): add WeightScope Rust quantization pipeline`
**Verdict:** **Does not compile. Pipeline is empty scaffolding.**

This review was verified by reading every cited file and running `cargo build` from the workspace root. Findings here are reproducible — they are not a delegated summary.

---

## Build status

```
$ cd quantization/quant-rs && cargo build
error: duplicate key
  --> main\quant-cli\Cargo.toml:23:13
   |
23 | safetensors.workspace = true
   |             ^^^^^^^^^
error: failed to load manifest for workspace member
       `...\quant-rs\main\quant-cli`
```

`safetensors.workspace = true` is declared at both `main/quant-cli/Cargo.toml:16` and `:23`. The workspace cannot resolve manifests, so nothing builds — not even unrelated crates.

---

## Missing implementations

The CLI binary `main/quant-cli/src/main.rs:8-10` imports the following symbols:

```rust
use quant_api::{Quantizer, QuantFormat};
use quant_engine::DefaultQuantService;
use quant_eval::EvalService;
use quant_io::ModelIO;
```

A workspace-wide grep for the corresponding type definitions (`struct DefaultQuantService`, `struct EvalService`, `struct ModelIO`, `struct Quantizer`, `enum QuantFormat`) returns **zero hits**. Only call-site references exist.

Each crate's `lib.rs` re-exports only `saf::*`:

```rust
// main/quant-engine/src/lib.rs (identical in quant-api, quant-eval, quant-io, quant-packer)
mod api;
mod core;
mod gateway;
mod saf;

pub use saf::*;
```

And every `saf/facade.rs` is identical boilerplate with empty bodies:

```rust
// main/quant-engine/src/saf/facade.rs:30-31 (same in all five crates)
pub fn quantize() {}
pub fn dequantize() {}
```

The CLI's `service.quantize(&weight)?` (`main/quant-cli/src/main.rs:111`) and `service.dequantize(&quantized)?` (`:114`) cannot compile against the published API of those crates. There is no actual quantization, packing, IO, or evaluation logic in the tree.

---

## Fake `DefaultService` boilerplate

`main/quant-{api,engine,eval,io,packer}/src/core/service/default_service.rs` are all the same file with the crate name swapped in the `println!`:

```rust
impl Service for DefaultService {
    fn execute(&self, config: &Config) -> Result<(), Error> {
        if config.verbose {
            println!("[crates/quant-engine] executing with verbose=true");
        }
        Ok(())
    }
}
```

The accompanying tests assert `is_ok()` on a function that always returns `Ok(())`. They cannot fail. Per the global metaprompt §1: *"A test that cannot fail is not a test."* Five crates × three tests = 15 trophy tests across the workspace.

---

## Content-correctness gap

Even if the workspace built, no test verifies that:

- A quantize → dequantize round trip reconstructs FP32 weights within tolerance
- NF4 / Int8 / Q4_0 bit-packing matches the reference layout
- `--protect-report` actually skips the listed layers in the output safetensors

This violates the user's standing rule (`feedback_content_correctness_required`): no quantization format may be claimed to "work" until output content is verified against a known-correct expectation.

---

## Concerns to address before re-claiming "pipeline"

| # | Issue | File | Severity |
|---|---|---|---|
| 1 | Workspace fails to load manifests | `main/quant-cli/Cargo.toml:16,23` | Blocker |
| 2 | `DefaultQuantService` referenced but never defined | `main/quant-cli/src/main.rs:8`, `main/sweetspot-finder/src/main.rs:6` | Blocker |
| 3 | `EvalService`, `ModelIO`, `Quantizer`, `QuantFormat` referenced but never defined | same | Blocker |
| 4 | All five SAF facades export empty `quantize()` / `dequantize()` | `main/*/src/saf/facade.rs:30-31` | High |
| 5 | All five `DefaultService` impls are identical println stubs with cannot-fail tests | `main/*/src/core/service/default_service.rs` | High |
| 6 | No round-trip verification of any quant format | (none exists) | High |
| 7 | Generic naming: `DefaultService::execute()` carries no signal | `main/*/src/core/service/default_service.rs:18-24` | Medium |

---

## Recommended next actions, in order

1. **Either revert `9a5bbd5` or stop describing the workspace as a "Rust quantization pipeline"** in commit messages, READMEs, and roadmap docs. The current state is incompatible with both claims.
2. Fix the duplicate `safetensors` dep so `cargo metadata` succeeds.
3. Implement the missing structs in their respective crates' `core/`, expose them through `lib.rs` (not via `saf::*` only), and rename `DefaultService::execute` to a verb-noun pair that describes the operation (`QuantizationService::quantize`).
4. Add a `--verify` mode to the CLI that asserts cosine similarity > 0.99 (NF4 / Int8) and > 0.95 (Q4_0) on a real tensor before any release tag.
5. Delete the cannot-fail `DefaultService` tests; replace with tests that load a known FP32 tensor, quantize it, dequantize it, and assert numeric tolerance.
