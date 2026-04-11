//! Integration tests for quantization strategy builder via QuantConfig.

use swe_ml_tensor::{quant_config_q8_all, quant_config_set_min_dim, quant_config_min_dim, quant_config_attention, QuantTarget};

/// @covers: quant_config_set_min_dim
#[test]
fn test_quant_config_builder_pattern_set_min_dim() {
    let mut c = quant_config_q8_all();
    quant_config_set_min_dim(&mut c, 2048);
    assert_eq!(quant_config_min_dim(&c), 2048);
    assert_eq!(quant_config_attention(&c), QuantTarget::Q8_0);
}
