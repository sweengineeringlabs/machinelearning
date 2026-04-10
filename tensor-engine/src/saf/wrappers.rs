// SAF wrapper functions — standalone functions that delegate to api traits.
// These are the public surface for consumers who don't use trait objects directly.

use crate::api::traits::TensorOps;
use crate::api::error::TensorResult;
use super::types::Tensor;

/// Returns the shape of a tensor as a slice.
pub fn tensor_shape(t: &Tensor) -> &[usize] {
    TensorOps::shape(t)
}

/// Returns the dtype of a tensor.
pub fn tensor_dtype(t: &Tensor) -> crate::api::dtype::DType {
    TensorOps::dtype(t)
}

/// Matrix multiply two tensors.
pub fn tensor_matmul(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    TensorOps::matmul(a, b)
}

/// Element-wise add two tensors.
pub fn tensor_add(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    TensorOps::add(a, b)
}

/// Softmax along a dimension.
pub fn tensor_softmax(t: &Tensor, dim: i64) -> TensorResult<Tensor> {
    TensorOps::softmax(t, dim)
}

/// Apply a runtime configuration globally.
///
/// Configures rayon thread pool, faer parallelism, and per-op thresholds.
pub fn apply_runtime_config(config: &super::types::RuntimeConfig) -> TensorResult<()> {
    config.apply_inner()
}

/// Warm up the rayon thread pool by forcing all threads to wake and do work.
pub fn warmup_thread_pool() {
    crate::core::runtime::runtime_config::RuntimeConfig::warmup_thread_pool();
}

/// Detect available SIMD instruction sets.
pub fn detect_simd() -> &'static str {
    crate::core::runtime::runtime_config::RuntimeConfig::detect_simd()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: tensor_shape
    #[test]
    fn test_tensor_shape_returns_correct_dims() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(tensor_shape(&t), &[2, 3]);
    }

    /// @covers: tensor_dtype
    #[test]
    fn test_tensor_dtype_returns_f32_for_default() {
        let t = Tensor::zeros(vec![2]);
        assert_eq!(tensor_dtype(&t), crate::api::dtype::DType::F32);
    }

    /// @covers: tensor_matmul
    #[test]
    fn test_tensor_matmul_correct_shape() {
        let a = Tensor::randn(vec![2, 3]);
        let b = Tensor::randn(vec![3, 4]);
        let c = tensor_matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
    }

    /// @covers: tensor_add
    #[test]
    fn test_tensor_add_sums() {
        let a = Tensor::ones(vec![3]);
        let b = Tensor::ones(vec![3]);
        let c = tensor_add(&a, &b).unwrap();
        assert_eq!(c.as_slice_f32().unwrap(), &[2.0, 2.0, 2.0]);
    }

    /// @covers: tensor_softmax
    #[test]
    fn test_tensor_softmax_sums_to_one() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let s = tensor_softmax(&t, -1).unwrap();
        let sum: f32 = s.as_slice_f32().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    /// @covers: apply_runtime_config
    #[test]
    fn test_apply_runtime_config_succeeds() {
        let config = super::super::types::RuntimeConfig::default();
        let result = apply_runtime_config(&config);
        assert!(result.is_ok());
    }

    /// @covers: detect_simd
    #[test]
    fn test_detect_simd_returns_nonempty() {
        let simd = detect_simd();
        assert!(!simd.is_empty());
    }

    /// @covers: warmup_thread_pool
    #[test]
    fn test_warmup_thread_pool_does_not_panic() {
        warmup_thread_pool();
    }
}
