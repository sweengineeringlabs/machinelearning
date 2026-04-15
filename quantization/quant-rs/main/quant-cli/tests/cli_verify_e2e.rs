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

/// Build a one-tensor safetensors file in `dir` and return its path.
/// Sin-wave fixture matches the quant-engine regression test so the
/// expected cosine bound (> 0.99 under Int8 block-32) is already proven.
fn write_sin_wave_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("in.safetensors");
    let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let view = TensorView::new(Dtype::F32, vec![4, 64], &bytes).expect("view");
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert("sin_wave".to_string(), view);
    serialize_to_file(&tensors, &None, &path).expect("write safetensors");
    path
}

#[test]
fn test_cli_verify_passes_for_smooth_signal() {
    let tmp = TempDir::new().expect("tmp");
    let in_path = write_sin_wave_fixture(tmp.path());
    let out_path = tmp.path().join("out.safetensors");

    let bin = env!("CARGO_BIN_EXE_quant-cli");
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

/// Cosine is mathematically bounded by [-1, 1]; a threshold of 1.01 is
/// unsatisfiable for any real quantization. Exercising this path proves
/// the CLI actually aborts (non-zero exit) and names the offending
/// tensor on stderr — i.e. `--verify` is not a silent no-op.
#[test]
fn test_cli_verify_aborts_when_threshold_is_unsatisfiable() {
    let tmp = TempDir::new().expect("tmp");
    let in_path = write_sin_wave_fixture(tmp.path());

    let bin = env!("CARGO_BIN_EXE_quant-cli");
    let output = Command::new(bin)
        .args([
            "--input",
            in_path.to_str().unwrap(),
            "--format",
            "int8",
            "--block-size",
            "32",
            "--verify",
            "--verify-threshold",
            "1.01",
        ])
        .output()
        .expect("spawn cli");

    assert!(
        !output.status.success(),
        "expected non-zero exit when threshold is unsatisfiable, got: {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("sin_wave"),
        "expected the failing tensor name on stderr, got: {stderr}"
    );
    assert!(
        stderr.to_lowercase().contains("cosine"),
        "expected 'cosine' in the error output, got: {stderr}"
    );
}
