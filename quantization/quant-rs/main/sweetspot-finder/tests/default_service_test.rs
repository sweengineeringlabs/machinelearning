//! DefaultService integration tests for crates/sweetspot-finder.

use crates_sweetspot_finder::*;

#[test]
fn test_default_service_through_facade() {
    // Exercise the default service through the saf facade
    assert!(run().is_ok());
}
