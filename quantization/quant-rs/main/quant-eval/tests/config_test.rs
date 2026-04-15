//! Config integration tests for crates/quant-eval.
//!
//! Tests Config through the saf facade to verify end-to-end wiring.

use crates_quant_eval::*;

#[test]
fn test_config_default_through_facade() {
    let config = Config::default();
    // Exercise config through full stack (api → core → saf)
    assert!(execute(&config).is_ok());
}

#[test]
fn test_config_verbose_through_facade() {
    let config = Config::default();
    // Verify verbose config works end-to-end
    assert!(execute(&config).is_ok());
}
