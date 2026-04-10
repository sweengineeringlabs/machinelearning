//! E2E tests for OptProfile.

use tensor_engine::OptProfile;

/// @covers: OptProfile::runtime_config
#[test]
fn test_opt_profile_optimized_default_thresholds() {
    let cfg = OptProfile::Optimized.runtime_config();
    assert_eq!(cfg.softmax_par_threshold, 4096);
}

/// @covers: OptProfile::runtime_config
#[test]
fn test_opt_profile_baseline_max_thresholds() {
    let cfg = OptProfile::Baseline.runtime_config();
    assert_eq!(cfg.softmax_par_threshold, usize::MAX);
}

/// @covers: OptProfile::use_inplace_ops
#[test]
fn test_opt_profile_baseline_disables_inplace() {
    assert!(!OptProfile::Baseline.use_inplace_ops());
    assert!(OptProfile::Optimized.use_inplace_ops());
}
