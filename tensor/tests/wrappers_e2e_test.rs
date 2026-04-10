//! E2E tests for SAF wrapper functions.

use swe_ml_tensor::{
    Tensor, DType,
    tensor_shape, tensor_dtype, tensor_matmul, tensor_add, tensor_softmax,
    warmup_thread_pool, detect_simd,
};

/// @covers: tensor_shape
#[test]
fn test_tensor_shape_wrapper_returns_correct_dims() {
    let t = Tensor::zeros(vec![2, 3]);
    assert_eq!(tensor_shape(&t), &[2, 3]);
}

/// @covers: tensor_dtype
#[test]
fn test_tensor_dtype_wrapper_returns_f32() {
    let t = Tensor::zeros(vec![2]);
    assert_eq!(tensor_dtype(&t), DType::F32);
}

/// @covers: tensor_matmul
#[test]
fn test_tensor_matmul_wrapper_produces_correct_shape() {
    let a = Tensor::randn([2, 3]);
    let b = Tensor::randn([3, 5]);
    let c = tensor_matmul(&a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 5]);
}

/// @covers: tensor_add
#[test]
fn test_tensor_add_wrapper_sums_elementwise() {
    let a = Tensor::ones([3]);
    let b = Tensor::ones([3]);
    let c = tensor_add(&a, &b).unwrap();
    let data = c.as_slice_f32().unwrap();
    assert!(data.iter().all(|&v| (v - 2.0).abs() < 1e-6));
}

/// @covers: tensor_softmax
#[test]
fn test_tensor_softmax_wrapper_sums_to_one() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let s = tensor_softmax(&t, -1).unwrap();
    let data = s.as_slice_f32().unwrap();
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}, expected 1.0");
}

/// @covers: detect_simd
#[test]
fn test_detect_simd_wrapper_returns_known_value() {
    let simd = detect_simd();
    let known = ["AVX2", "SSE2", "NEON", "scalar"];
    assert!(known.contains(&simd), "unexpected SIMD: {}", simd);
}

/// @covers: warmup_thread_pool
#[test]
fn test_warmup_thread_pool_does_not_panic() {
    warmup_thread_pool();
}
