//! E2E tests for RuntimeConfig.

use tensor_engine::{RuntimeConfig, detect_simd};

/// @covers: RuntimeConfig::default
#[test]
fn test_runtime_config_default_thresholds() {
    let config = RuntimeConfig::default();
    assert_eq!(config.num_threads, 0);
    assert_eq!(config.softmax_par_threshold, 4096);
    assert_eq!(config.batched_matmul_par_threshold, 4096);
    assert_eq!(config.gemv_par_threshold, 4096);
}

/// @covers: detect_simd
#[test]
fn test_detect_simd_returns_known_value() {
    let simd = detect_simd();
    let known = ["AVX2", "SSE2", "NEON", "scalar"];
    assert!(known.contains(&simd), "unexpected SIMD: {}", simd);
}
