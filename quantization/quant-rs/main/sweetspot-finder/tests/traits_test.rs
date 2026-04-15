//! Trait integration tests for crates/sweetspot-finder.

use crates_sweetspot_finder::*;

#[test]
fn test_service_trait_is_object_safe() {
    // Verify the Service trait can be used as a trait object
    fn _accept(_s: &dyn Service) {}
}
