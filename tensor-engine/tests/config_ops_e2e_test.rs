//! E2E tests for ConfigOps trait.

use tensor_engine::{ConfigOps, RuntimeConfig};

/// @covers: ConfigOps::apply
#[test]
fn test_config_ops_apply_via_default_config() {
    let config = RuntimeConfig::default();
    // auto-detect threads; should not panic
    let result = config.apply();
    assert!(result.is_ok());
}
