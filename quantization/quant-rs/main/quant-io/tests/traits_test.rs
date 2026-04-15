//! Trait integration tests for crates/quant-io.

use crates_quant_io::*;

#[test]
fn test_service_trait_is_object_safe() {
    // Verify the Service trait can be used as a trait object
    fn _accept(_s: &dyn Service) {}
}
