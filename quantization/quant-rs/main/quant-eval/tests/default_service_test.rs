//! DefaultService integration tests for crates/quant-eval.

use crates_quant_eval::*;

#[test]
fn test_default_service_through_facade() {
    // Exercise the default service through the saf facade
    assert!(run().is_ok());
}
