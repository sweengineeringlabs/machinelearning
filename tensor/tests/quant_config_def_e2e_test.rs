//! E2E tests for QuantConfig definition.

use swe_ml_tensor::{quant_config_q8_all, quant_config_none, quant_config_attention, QuantTarget};

#[test]
fn test_quant_config_def_q8_and_none_differ() {
    let q8 = quant_config_q8_all();
    let none = quant_config_none();
    assert_eq!(quant_config_attention(&q8), QuantTarget::Q8_0);
    assert_eq!(quant_config_attention(&none), QuantTarget::None);
}
