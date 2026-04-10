//! Tests for core module.
use training_engine::*;

#[test]
fn test_module_accessible() {
    let _ = Tensor::zeros(vec![1]);
}
