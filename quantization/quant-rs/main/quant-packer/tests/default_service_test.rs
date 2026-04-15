//! DefaultService integration tests for crates/quant-packer.

use crates_quant_packer::*;

#[test]
fn test_default_service_through_facade() {
    // Exercise the default service through the saf facade
    assert!(run().is_ok());
}
