//! Integration tests for the ConfigOps trait.

use tensor_engine::ConfigOps;

/// @covers: ConfigOps::apply
#[test]
fn test_config_ops_apply_via_opt_profile_optimized() {
    let profile = tensor_engine::OptProfile::Optimized;
    let result = profile.apply();
    assert!(result.is_ok());
}
