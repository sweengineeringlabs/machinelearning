//! E2E tests for the QuantTarget enum.

use swe_ml_tensor::QuantTarget;

/// @covers: QuantTarget (Debug)
#[test]
fn test_quant_target_debug_shows_variant() {
    assert_eq!(format!("{:?}", QuantTarget::Q8_0), "Q8_0");
    assert_eq!(format!("{:?}", QuantTarget::None), "None");
    assert_eq!(format!("{:?}", QuantTarget::F16), "F16");
}

/// @covers: QuantTarget (Clone)
#[test]
fn test_quant_target_clone_preserves_equality() {
    let t = QuantTarget::Q4_0;
    assert_eq!(t, t.clone());
}
