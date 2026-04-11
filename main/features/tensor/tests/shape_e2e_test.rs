//! E2E tests for the Shape type.

use swe_ml_tensor::Shape;

/// @covers: Shape::new
#[test]
fn test_shape_new_stores_dims() {
    let s = Shape::new(vec![2, 3, 4]);
    assert_eq!(s.dims(), &[2, 3, 4]);
}

/// @covers: Shape::ndim
#[test]
fn test_shape_ndim_returns_dimension_count() {
    let s = Shape::new(vec![2, 3]);
    assert_eq!(s.ndim(), 2);
}

/// @covers: Shape::numel
#[test]
fn test_shape_numel_returns_product() {
    let s = Shape::new(vec![2, 3, 4]);
    assert_eq!(s.numel(), 24);
}

/// @covers: Shape::scalar
#[test]
fn test_shape_scalar_has_zero_dims() {
    let s = Shape::scalar();
    assert!(s.is_scalar());
    assert_eq!(s.ndim(), 0);
}

/// @covers: Shape::broadcast_with
#[test]
fn test_shape_broadcast_compatible() {
    let a = Shape::new(vec![2, 1, 4]);
    let b = Shape::new(vec![3, 4]);
    let c = a.broadcast_with(&b);
    assert_eq!(c, Some(Shape::new(vec![2, 3, 4])));
}

/// @covers: Shape::broadcast_with
#[test]
fn test_shape_broadcast_incompatible_returns_none() {
    let a = Shape::new(vec![2, 3]);
    let b = Shape::new(vec![4, 5]);
    assert!(a.broadcast_with(&b).is_none());
}
