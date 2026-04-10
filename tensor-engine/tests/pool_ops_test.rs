//! Integration tests for the PoolOps trait via TensorPool.

/// @covers: tensor_engine::create_tensor_builder
#[test]
fn test_tensor_builder_via_public_api() {
    let t = tensor_engine::create_tensor_builder()
        .shape(vec![3, 4])
        .zeros()
        .unwrap();
    assert_eq!(t.shape(), &[3, 4]);
}
