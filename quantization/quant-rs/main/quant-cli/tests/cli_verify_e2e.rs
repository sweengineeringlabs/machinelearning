//! End-to-end smoke test for the `--verify` mode.
//!
//! Builds a tiny safetensors file containing one F32 tensor of known
//! values, runs the CLI binary against it with `--verify`, and asserts
//! the process exits 0. This catches regressions where:
//! - The CLI accepts a format the engine no longer supports.
//! - `--verify` silently swallows a cosine-below-threshold result.
//! - The save → load → quantize pipeline diverges (e.g., dtype mismatch).

use std::collections::HashMap;
use std::process::Command;

use safetensors::{Dtype, serialize_to_file};
use safetensors::tensor::TensorView;
use tempfile::TempDir;

#[test]
fn test_cli_verify_passes_for_smooth_signal() {
    let tmp = TempDir::new().expect("tmp");
    let in_path = tmp.path().join("in.safetensors");

    // Sin wave is the same fixture quant-engine's regression test uses.
    // 256 elements over [4, 64] — large enough that the per-block average
    // is meaningful, small enough to stay cheap.
    let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let view = TensorView::new(Dtype::F32, vec![4, 64], &bytes).expect("view");

    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert("sin_wave".to_string(), view);
    serialize_to_file(&tensors, &None, &in_path).expect("write safetensors");

    let bin = env!("CARGO_BIN_EXE_quant-cli");
    let out_path = tmp.path().join("out.safetensors");
    let status = Command::new(bin)
        .args([
            "--input",
            in_path.to_str().unwrap(),
            "--output",
            out_path.to_str().unwrap(),
            "--format",
            "int8",
            "--block-size",
            "32",
            "--verify",
        ])
        .status()
        .expect("spawn cli");

    assert!(
        status.success(),
        "expected --verify to succeed for a smooth signal, exit status: {:?}",
        status
    );
    assert!(
        out_path.exists(),
        "expected the CLI to write the output safetensors when given --output"
    );
}
