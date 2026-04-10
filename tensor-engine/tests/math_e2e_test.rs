//! E2E tests for tensor math operations (matmul, add, softmax, etc.).
//! Covers: faer dependency (used for matrix multiplication kernels).

// faer is used transitively via Tensor::matmul for optimized BLAS operations
use tensor_engine::{Tensor, DType};

/// @covers: Tensor::matmul
#[test]
fn test_matmul_identity_preserves_values() {
    let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
}

/// @covers: Tensor::add
#[test]
fn test_add_elementwise() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0]);
}

/// @covers: Tensor::softmax
#[test]
fn test_softmax_output_sums_to_one() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let s = t.softmax(-1).unwrap();
    let data = s.as_slice_f32().unwrap();
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
}

/// @covers: Tensor::matmul
#[test]
fn test_matmul_dimension_mismatch_returns_error() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0], vec![1, 2]).unwrap();
    assert!(a.matmul(&b).is_err());
}
