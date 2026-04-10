use tensor_engine::{Tensor, DType};

#[test]
fn test_create_zeros_returns_correct_shape_and_dtype() {
    let t = Tensor::zeros([2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
}

#[test]
fn test_create_ones_returns_all_ones() {
    let t = Tensor::ones([4]);
    let data = t.as_slice_f32().unwrap();
    assert!(data.iter().all(|&v| v == 1.0));
}

#[test]
fn test_matmul_produces_correct_output_shape() {
    let a = Tensor::randn([2, 3]);
    let b = Tensor::randn([3, 5]);
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 5]);
}

#[test]
fn test_add_broadcasts_scalar_to_vector() {
    let a = Tensor::ones([3]);
    let b = Tensor::ones([3]);
    let c = a.add(&b).unwrap();
    let data = c.as_slice_f32().unwrap();
    assert!(data.iter().all(|&v| (v - 2.0).abs() < 1e-6));
}

#[test]
fn test_to_f16_changes_dtype() {
    let t = Tensor::ones([4]);
    let f16 = t.to_f16().unwrap();
    assert_eq!(f16.dtype(), DType::F16);
}

#[test]
fn test_softmax_output_sums_to_one() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let s = t.softmax(-1).unwrap();
    let data = s.as_slice_f32().unwrap();
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}, expected 1.0");
}
