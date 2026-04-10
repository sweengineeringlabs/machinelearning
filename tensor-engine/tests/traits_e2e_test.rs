//! E2E tests for api traits re-exports.

use tensor_engine::{Tensor, TensorPool, DType};

/// @covers: TensorOps (via Tensor)
#[test]
fn test_tensor_ops_shape_via_public_api() {
    let t = Tensor::zeros(vec![2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
}

/// @covers: PoolOps (via TensorPool)
#[test]
fn test_pool_ops_via_public_api() {
    let mut pool = TensorPool::new(4);
    assert!(pool.is_empty());
    let buf = pool.get(64);
    assert_eq!(buf.len(), 64);
    pool.put(buf);
    assert_eq!(pool.len(), 1);
}
