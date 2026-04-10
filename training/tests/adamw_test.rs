//! Tests for core module.
use swe_ml_training::*;

#[test]
fn test_module_accessible() {
    let _ = Tensor::zeros(vec![1]);
}
