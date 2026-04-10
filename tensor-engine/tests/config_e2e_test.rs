//! E2E tests for core/runtime/quant/config (QuantConfig factory methods).

use tensor_engine::{quant_config_from_toml_file, quant_config_attention, QuantTarget};

#[test]
fn test_config_from_nonexistent_toml_uses_defaults() {
    let c = quant_config_from_toml_file(std::path::Path::new("/no/such/config.toml"));
    assert_eq!(quant_config_attention(&c), QuantTarget::Q8_0);
}
