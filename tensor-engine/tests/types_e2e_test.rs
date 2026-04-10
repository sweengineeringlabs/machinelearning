//! E2E tests for saf/types re-exports.

use tensor_engine::{Tensor, Shape};

/// @covers: Tensor (re-export)
#[test]
fn test_tensor_reexport_constructible() {
    let t = Tensor::zeros(vec![2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
}

/// @covers: Shape (re-export)
#[test]
fn test_shape_reexport_constructible() {
    let s = Shape::new(vec![4, 5]);
    assert_eq!(s.ndim(), 2);
}
