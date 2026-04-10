//! E2E tests for QuantOps trait via QuantStrategy.

use tensor_engine::{QuantStrategy, QuantTarget};

/// @covers: QuantOps (via QuantStrategy)
#[test]
fn test_quant_strategy_none_via_public_api() {
    let s = QuantStrategy::none();
    assert_eq!(s.attention, QuantTarget::None);
}

/// @covers: QuantOps (via QuantStrategy)
#[test]
fn test_quant_strategy_from_toml_via_public_api() {
    let toml = "[quantization]\nattention = \"q4_0\"\n";
    let s = QuantStrategy::from_toml(toml).unwrap();
    assert_eq!(s.attention, QuantTarget::Q4_0);
}
