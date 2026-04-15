//! DefaultService integration tests for crates/quant-engine.

use crates_quant_engine::*;

#[test]
fn test_default_service_through_facade() {
    // Exercise the default service through the saf facade
    assert!(run().is_ok());
}
