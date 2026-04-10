//! Integration tests for quantization strategy via QuantConfig.

use swe_ml_tensor::{quant_config_from_toml_file, quant_config_q8_all, quant_config_attention, quant_config_feed_forward, quant_config_output, quant_config_moe, quant_config_gate, quant_config_min_dim, QuantTarget};

/// @covers: quant_config_from_toml_file
#[test]
fn test_quant_config_from_nonexistent_file_uses_defaults() {
    let c = quant_config_from_toml_file(std::path::Path::new("/no/such/file.toml"));
    assert_eq!(quant_config_attention(&c), QuantTarget::Q8_0);
    assert_eq!(quant_config_feed_forward(&c), QuantTarget::Q8_0);
}

/// @covers: quant_config_q8_all
#[test]
fn test_quant_config_all_accessors_consistent() {
    let c = quant_config_q8_all();
    assert_eq!(quant_config_attention(&c), QuantTarget::Q8_0);
    assert_eq!(quant_config_feed_forward(&c), QuantTarget::Q8_0);
    assert_eq!(quant_config_output(&c), QuantTarget::Q8_0);
    assert_eq!(quant_config_moe(&c), QuantTarget::Q8_0);
    assert_eq!(quant_config_gate(&c), QuantTarget::Q8_0);
    assert_eq!(quant_config_min_dim(&c), 0);
}
