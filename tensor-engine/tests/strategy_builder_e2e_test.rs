//! E2E tests for QuantStrategyBuilder.

use tensor_engine::{QuantStrategyBuilder, QuantTarget};

/// @covers: QuantStrategyBuilder::new
#[test]
fn test_strategy_builder_new_defaults_to_q8() {
    let s = QuantStrategyBuilder::new().build();
    assert_eq!(s.attention, QuantTarget::Q8_0);
    assert_eq!(s.feed_forward, QuantTarget::Q8_0);
}

/// @covers: QuantStrategyBuilder::attention
#[test]
fn test_strategy_builder_attention_override() {
    let s = QuantStrategyBuilder::new()
        .attention(QuantTarget::None)
        .build();
    assert_eq!(s.attention, QuantTarget::None);
    assert_eq!(s.feed_forward, QuantTarget::Q8_0);
}

/// @covers: QuantStrategyBuilder::min_dim
#[test]
fn test_strategy_builder_min_dim_override() {
    let s = QuantStrategyBuilder::new()
        .min_dim(2048)
        .build();
    assert_eq!(s.min_dim, 2048);
}
