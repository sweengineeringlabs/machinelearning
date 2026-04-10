//! Integration tests for bytemuck-dependent functions in storage_def.

use tensor_engine::{Tensor, f32_vec_to_bytes, f32_slice_to_bytes};
use bytemuck;

/// @covers: f32_vec_to_bytes
#[test]
fn test_f32_vec_to_bytes_roundtrip() {
    let original = vec![1.0f32, 2.0, 3.0, -4.5];
    let bytes = f32_vec_to_bytes(original.clone());
    assert_eq!(bytes.len(), original.len() * 4);
    let recovered: &[f32] = bytemuck::cast_slice(&bytes);
    assert_eq!(recovered, &original);
}

/// @covers: f32_vec_to_bytes
#[test]
fn test_f32_vec_to_bytes_empty() {
    let bytes = f32_vec_to_bytes(vec![]);
    assert!(bytes.is_empty());
}

/// @covers: f32_slice_to_bytes
#[test]
fn test_f32_slice_to_bytes_correct_length() {
    let data = [1.0f32, 2.0, 3.0];
    let bytes = f32_slice_to_bytes(&data);
    assert_eq!(bytes.len(), 12); // 3 * 4 bytes
}

/// @covers: f32_slice_to_bytes
#[test]
fn test_f32_slice_to_bytes_empty_slice() {
    let data: [f32; 0] = [];
    let bytes = f32_slice_to_bytes(&data);
    assert!(bytes.is_empty());
}

/// @covers: Tensor::as_slice_f32
#[test]
fn test_tensor_as_slice_f32_returns_correct_values() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let slice = t.as_slice_f32().unwrap();
    assert_eq!(slice, &[1.0, 2.0, 3.0]);
}

/// @covers: Tensor::as_slice_f32
#[test]
fn test_tensor_as_slice_f32_zeros() {
    let t = Tensor::zeros(vec![4]);
    let slice = t.as_slice_f32().unwrap();
    assert_eq!(slice, &[0.0, 0.0, 0.0, 0.0]);
}
