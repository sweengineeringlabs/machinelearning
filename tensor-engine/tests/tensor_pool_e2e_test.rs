//! E2E tests for TensorPool via create_tensor_builder.
//! TensorPool is pub(crate) so we test it indirectly.

use tensor_engine::create_tensor_builder;

#[test]
fn test_tensor_pool_indirect_via_builder() {
    // Builder uses pool internally; this validates the allocation path
    let t = create_tensor_builder().shape(vec![4, 4]).ones().unwrap();
    let data = t.as_slice_f32().unwrap();
    assert!(data.iter().all(|&v| v == 1.0));
}
