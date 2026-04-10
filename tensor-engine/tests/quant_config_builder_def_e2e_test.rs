//! E2E tests for QuantConfigBuilder.

use tensor_engine::{quant_config_q8_all, quant_config_set_min_dim, quant_config_min_dim};

#[test]
fn test_quant_config_builder_set_min_dim() {
    let mut c = quant_config_q8_all();
    quant_config_set_min_dim(&mut c, 512);
    assert_eq!(quant_config_min_dim(&c), 512);
}
