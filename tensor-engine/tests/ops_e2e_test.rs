//! E2E tests for quant ops trait.

use tensor_engine::{quant_config_q8_all, quant_config_none, quant_config_attention, QuantTarget};

#[test]
fn test_quant_ops_q8_all_via_config() {
    let c = quant_config_q8_all();
    assert_eq!(quant_config_attention(&c), QuantTarget::Q8_0);
}

#[test]
fn test_quant_ops_none_via_config() {
    let c = quant_config_none();
    assert_eq!(quant_config_attention(&c), QuantTarget::None);
}
