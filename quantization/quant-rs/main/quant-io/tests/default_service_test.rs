//! DefaultService integration tests for crates/quant-io.

use crates_quant_io::*;

#[test]
fn test_default_service_through_facade() {
    // Exercise the default service through the saf facade
    assert!(run().is_ok());
}
