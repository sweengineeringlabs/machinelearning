//! Integration tests for TensorBuilder via the public API.

use swe_ml_tensor::{create_tensor_builder, Tensor, DType};

/// @covers: create_tensor_builder
#[test]
fn test_builder_zeros_via_public_api() {
    let t = create_tensor_builder()
        .shape(vec![2, 3])
        .zeros()
        .unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    let data = t.as_slice_f32().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}

/// @covers: create_tensor_builder
#[test]
fn test_builder_ones_via_public_api() {
    let t = create_tensor_builder()
        .shape(vec![4])
        .ones()
        .unwrap();
    let data = t.as_slice_f32().unwrap();
    assert!(data.iter().all(|&v| v == 1.0));
}

/// @covers: create_tensor_builder
#[test]
fn test_builder_from_data_via_public_api() {
    let t = create_tensor_builder()
        .shape(vec![2, 2])
        .from_data(vec![1.0, 2.0, 3.0, 4.0])
        .unwrap();
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// @covers: create_tensor_builder
#[test]
fn test_builder_without_shape_fails() {
    let result = create_tensor_builder().zeros();
    assert!(result.is_err());
}
