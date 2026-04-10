//! E2E tests for api/types re-exports.

use tensor_engine::{Tensor, Shape, TensorPool, TensorBuilder};

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

/// @covers: TensorPool (re-export)
#[test]
fn test_tensor_pool_reexport_constructible() {
    let pool = TensorPool::new(8);
    assert!(pool.is_empty());
}

/// @covers: TensorBuilder (re-export)
#[test]
fn test_tensor_builder_reexport_constructible() {
    let t = TensorBuilder::new()
        .shape(vec![2, 2])
        .zeros()
        .unwrap();
    assert_eq!(t.shape(), &[2, 2]);
}
