//! Integration tests for the TensorOps trait implementation on Tensor.

use tensor_engine::{Tensor, DType};

/// @covers: TensorOps::shape
#[test]
fn test_tensor_ops_shape_returns_correct_dimensions() {
    let t = Tensor::zeros(vec![3, 4]);
    assert_eq!(t.shape(), &[3, 4]);
}

/// @covers: TensorOps::dtype
#[test]
fn test_tensor_ops_dtype_returns_f32_for_default() {
    let t = Tensor::ones(vec![2]);
    assert_eq!(t.dtype(), DType::F32);
}

/// @covers: TensorOps::matmul
#[test]
fn test_tensor_ops_matmul_produces_correct_result() {
    let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    // Identity * B = B
    assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
}

/// @covers: TensorOps::add
#[test]
fn test_tensor_ops_add_sums_elementwise() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0]);
}

/// @covers: TensorOps::softmax
#[test]
fn test_tensor_ops_softmax_sums_to_one() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let s = t.softmax(-1).unwrap();
    let data = s.as_slice_f32().unwrap();
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {}, expected 1.0", sum);
}

/// @covers: TensorOps::matmul
#[test]
fn test_tensor_ops_matmul_dimension_mismatch_returns_error() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0], vec![1, 2]).unwrap();
    assert!(a.matmul(&b).is_err());
}

/// @covers: TensorOps::add
#[test]
fn test_tensor_ops_add_with_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let b = Tensor::from_vec(vec![10.0], vec![1]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.get(&[0, 0]).unwrap(), 11.0);
    assert_eq!(c.get(&[1, 2]).unwrap(), 16.0);
}
