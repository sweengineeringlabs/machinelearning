//! Trait integration tests for crates/quant-packer.

use crates_quant_packer::*;

#[test]
fn test_service_trait_is_object_safe() {
    // Verify the Service trait can be used as a trait object
    fn _accept(_s: &dyn Service) {}
}
