//! E2E tests for OptProfile.

use swe_ml_tensor::OptProfile;

/// @covers: OptProfile::apply
#[test]
fn test_opt_profile_optimized_applies_without_error() {
    assert!(OptProfile::Optimized.apply().is_ok());
}

/// @covers: OptProfile::use_inplace_ops
#[test]
fn test_opt_profile_baseline_disables_inplace() {
    assert!(!OptProfile::Baseline.use_inplace_ops());
    assert!(OptProfile::Optimized.use_inplace_ops());
}

/// @covers: OptProfile::use_buffered_sampling
#[test]
fn test_opt_profile_baseline_disables_buffered_sampling() {
    assert!(!OptProfile::Baseline.use_buffered_sampling());
    assert!(OptProfile::Aggressive.use_buffered_sampling());
}
