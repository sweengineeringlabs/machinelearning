//! E2E tests for QuantStrategy and QuantStrategyBuilder.

use tensor_engine::{QuantStrategy, QuantStrategyBuilder, QuantTarget};

/// @covers: QuantStrategy::default
#[test]
fn test_quant_strategy_default_is_q8() {
    let s = QuantStrategy::default();
    assert_eq!(s.attention, QuantTarget::Q8_0);
    assert_eq!(s.feed_forward, QuantTarget::Q8_0);
    assert_eq!(s.output, QuantTarget::Q8_0);
}

/// @covers: QuantStrategy::none
#[test]
fn test_quant_strategy_none_skips_all() {
    let s = QuantStrategy::none();
    assert_eq!(s.attention, QuantTarget::None);
    assert_eq!(s.feed_forward, QuantTarget::None);
}

/// @covers: QuantStrategy::from_toml
#[test]
fn test_quant_strategy_from_toml_parses() {
    let toml = r#"
[quantization]
attention = "q8_0"
output = "none"
"#;
    let s = QuantStrategy::from_toml(toml).unwrap();
    assert_eq!(s.attention, QuantTarget::Q8_0);
    assert_eq!(s.output, QuantTarget::None);
}

/// @covers: QuantStrategyBuilder::build
#[test]
fn test_quant_strategy_builder_overrides() {
    let s = QuantStrategyBuilder::new()
        .attention(QuantTarget::None)
        .output(QuantTarget::F16)
        .build();
    assert_eq!(s.attention, QuantTarget::None);
    assert_eq!(s.output, QuantTarget::F16);
    assert_eq!(s.feed_forward, QuantTarget::Q8_0);
}
