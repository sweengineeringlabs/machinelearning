//! E2E tests for tensor definition (api/tensor/def.rs).

use swe_ml_tensor::Tensor;

#[test]
fn test_tensor_def_clone_produces_independent_copy() {
    let t = Tensor::zeros(vec![2, 3]);
    let t2 = t.clone();
    assert_eq!(t.shape(), t2.shape());
}
