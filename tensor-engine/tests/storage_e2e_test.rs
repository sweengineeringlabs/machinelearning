//! E2E tests for Storage and bytemuck-based byte conversions.
//! Covers: bytemuck dependency (f32 <-> byte slice conversions).

// bytemuck is used transitively via f32_vec_to_bytes / f32_slice_to_bytes
use tensor_engine::{Tensor, f32_vec_to_bytes, f32_slice_to_bytes, DType};

/// @covers: f32_vec_to_bytes
#[test]
fn test_f32_vec_to_bytes_roundtrip_preserves_data() {
    let original = vec![1.0f32, 2.0, 3.0, -4.5];
    let bytes = f32_vec_to_bytes(original.clone());
    assert_eq!(bytes.len(), 16, "4 f32s should produce 16 bytes");

    // Reconstruct via Tensor
    let t = Tensor::new(bytes, smallvec::smallvec![4usize], DType::F32);
    let recovered = t.as_slice_f32().unwrap();
    assert_eq!(recovered, &original[..]);
}

/// @covers: f32_slice_to_bytes
#[test]
fn test_f32_slice_to_bytes_returns_correct_length() {
    let data = [1.0f32, 2.0, 3.0];
    let bytes = f32_slice_to_bytes(&data);
    assert_eq!(bytes.len(), 12);
}

/// @covers: f32_vec_to_bytes
#[test]
fn test_f32_vec_to_bytes_empty_produces_empty() {
    let bytes = f32_vec_to_bytes(vec![]);
    assert!(bytes.is_empty());
}

/// @covers: Tensor::as_slice_f32
#[test]
fn test_tensor_as_slice_f32_uses_bytemuck_alignment() {
    let t = Tensor::from_vec(vec![42.0, 99.0], vec![2]).unwrap();
    let slice = t.as_slice_f32().unwrap();
    assert_eq!(slice, &[42.0, 99.0]);
    // Verify alignment: the pointer should be 4-byte aligned
    assert_eq!(slice.as_ptr() as usize % 4, 0, "f32 slice must be 4-byte aligned");
}

/// @covers: Storage::Owned
#[test]
fn test_storage_owned_via_tensor_from_vec() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    assert_eq!(t.shape(), &[3]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.as_slice_f32().unwrap(), &[1.0, 2.0, 3.0]);
}

/// @covers: Storage::Owned
#[test]
fn test_storage_owned_via_zeros() {
    let t = Tensor::zeros([2, 2]);
    let data = t.as_slice_f32().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}
