//! E2E tests for tensor view operations (reshape, transpose, slice, etc.).

use tensor_engine::{Tensor, Shape};

/// @covers: Tensor::reshape
#[test]
fn test_reshape_changes_shape_preserves_data() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let r = t.reshape(&[3, 2]).unwrap();
    assert_eq!(r.shape(), &[3, 2]);
}

/// @covers: Tensor::transpose
#[test]
fn test_transpose_swaps_dimensions() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let tr = t.transpose(-2, -1).unwrap();
    assert_eq!(tr.shape(), &[3, 2]);
}

/// @covers: Tensor::select
#[test]
fn test_select_reduces_dimension() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let selected = t.select(0, 1).unwrap();
    assert_eq!(selected.shape(), &[3]);
    assert_eq!(selected.to_vec(), vec![4.0, 5.0, 6.0]);
}

/// @covers: Tensor::slice
#[test]
fn test_slice_along_dimension() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let sliced = t.slice(1, 0, 2).unwrap();
    assert_eq!(sliced.shape(), &[2, 2]);
}

/// @covers: Tensor::broadcast_to
#[test]
fn test_broadcast_to_expands_dimensions() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let target = Shape::new(vec![2, 3]);
    let b = a.broadcast_to(&target).unwrap();
    assert_eq!(b.shape(), &[2, 3]);
}

/// @covers: Tensor::unsqueeze
#[test]
fn test_unsqueeze_adds_dimension() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let u = t.unsqueeze(0).unwrap();
    assert_eq!(u.shape(), &[1, 3]);
}
