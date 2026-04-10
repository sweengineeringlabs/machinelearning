//! E2E tests for TensorBuilder.

use tensor_engine::{TensorBuilder, DType};

/// @covers: TensorBuilder::zeros
#[test]
fn test_tensor_builder_zeros_correct_shape() {
    let t = TensorBuilder::new()
        .shape(vec![3, 4])
        .zeros()
        .unwrap();
    assert_eq!(t.shape(), &[3, 4]);
    assert_eq!(t.dtype(), DType::F32);
}

/// @covers: TensorBuilder::ones
#[test]
fn test_tensor_builder_ones_all_ones() {
    let t = TensorBuilder::new()
        .shape(vec![5])
        .ones()
        .unwrap();
    let data = t.as_slice_f32().unwrap();
    assert!(data.iter().all(|&v| v == 1.0));
}

/// @covers: TensorBuilder::from_data
#[test]
fn test_tensor_builder_from_data() {
    let t = TensorBuilder::new()
        .shape(vec![2, 2])
        .from_data(vec![1.0, 2.0, 3.0, 4.0])
        .unwrap();
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// @covers: TensorBuilder::zeros
#[test]
fn test_tensor_builder_no_shape_returns_error() {
    let result = TensorBuilder::new().zeros();
    assert!(result.is_err());
}
