//! Integration tests for the ConfigOps trait.

use swe_ml_tensor::ConfigOps;

/// @covers: ConfigOps::apply
#[test]
fn test_config_ops_apply_via_opt_profile_optimized() {
    let profile = swe_ml_tensor::OptProfile::Optimized;
    let result = profile.apply();
    assert!(result.is_ok());
}
