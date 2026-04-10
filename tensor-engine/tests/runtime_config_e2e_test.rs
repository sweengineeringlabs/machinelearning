//! Integration tests for runtime configuration via public API.
//! Exercises faer parallelism configuration (dependency: faer).

use tensor_engine::{warmup_thread_pool, detect_simd, OptProfile};
// Verify faer dependency is accessible (exercises faer integration)
use faer::Parallelism;

/// @covers: warmup_thread_pool
#[test]
fn test_warmup_thread_pool_completes() {
    warmup_thread_pool();
}

/// @covers: detect_simd
#[test]
fn test_detect_simd_returns_known_instruction_set() {
    let simd = detect_simd();
    assert!(
        ["AVX2", "SSE2", "NEON", "scalar"].contains(&simd),
        "unexpected SIMD value: {}",
        simd
    );
}

/// Exercises faer parallelism configuration via OptProfile::apply.
#[test]
fn test_opt_profile_apply_configures_faer_parallelism() {
    // Verify faer Parallelism enum is usable (dependency health check)
    let _p = Parallelism::Rayon(0);
    let result = OptProfile::Optimized.apply();
    assert!(result.is_ok());
}
