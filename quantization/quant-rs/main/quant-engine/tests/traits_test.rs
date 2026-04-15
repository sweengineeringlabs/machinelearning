//! Trait integration tests for crates/quant-engine.

use crates_quant_engine::*;

#[test]
fn test_service_trait_is_object_safe() {
    // Verify the Service trait can be used as a trait object
    fn _accept(_s: &dyn Service) {}
}
