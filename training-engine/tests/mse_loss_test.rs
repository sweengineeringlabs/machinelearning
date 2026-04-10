//! Tests for MSELoss core module.
use training_engine::*;

/// @covers: MSELoss::forward
#[test]
fn test_forward_computes_mean_squared_error() {
    let loss = MSELoss::new();
    let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
    let tgt = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
    let result = loss.forward(&pred, &tgt).unwrap();
    assert!((result.to_vec()[0] - 4.0).abs() < 1e-6);
}
