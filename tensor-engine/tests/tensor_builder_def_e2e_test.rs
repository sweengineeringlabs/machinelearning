//! E2E tests for TensorBuilder definition.

use tensor_engine::create_tensor_builder;

#[test]
fn test_tensor_builder_def_constructs_tensor() {
    let t = create_tensor_builder().shape(vec![3]).zeros().unwrap();
    assert_eq!(t.shape(), &[3]);
}
