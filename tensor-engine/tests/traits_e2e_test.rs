//! E2E tests for api traits re-exports.

use tensor_engine::{Tensor, DType};

/// @covers: TensorOps (via Tensor)
#[test]
fn test_tensor_ops_shape_via_public_api() {
    let t = Tensor::zeros(vec![2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
}
