//! Integration tests for quant module (QuantConfig and QuantTarget).

use swe_ml_tensor::{
    quant_config_q8_all, quant_config_none,
    quant_config_attention, quant_config_feed_forward, quant_config_output,
    quant_config_moe, quant_config_gate,
    quant_config_min_dim, quant_config_set_min_dim,
    QuantTarget,
};

/// @covers: quant_config_q8_all
#[test]
fn test_quant_config_q8_all_via_public_api() {
    let c = quant_config_q8_all();
    assert_eq!(quant_config_attention(&c), QuantTarget::Q8_0);
    assert_eq!(quant_config_feed_forward(&c), QuantTarget::Q8_0);
}

/// @covers: quant_config_none
#[test]
fn test_quant_config_none_via_public_api() {
    let c = quant_config_none();
    assert_eq!(quant_config_attention(&c), QuantTarget::None);
}

/// @covers: quant_config_set_min_dim
#[test]
fn test_quant_config_set_min_dim_roundtrip() {
    let mut c = quant_config_q8_all();
    assert_eq!(quant_config_min_dim(&c), 0);
    quant_config_set_min_dim(&mut c, 1536);
    assert_eq!(quant_config_min_dim(&c), 1536);
}

/// @covers: quant_config_moe
#[test]
fn test_quant_config_q8_all_moe_is_q8() {
    let c = quant_config_q8_all();
    assert_eq!(quant_config_moe(&c), QuantTarget::Q8_0);
}

/// @covers: quant_config_gate
#[test]
fn test_quant_config_none_gate_is_none() {
    let c = quant_config_none();
    assert_eq!(quant_config_gate(&c), QuantTarget::None);
}
