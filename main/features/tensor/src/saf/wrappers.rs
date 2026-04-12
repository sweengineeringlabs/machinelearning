// SAF wrapper functions — standalone functions that delegate to api traits.
// These are the public surface for consumers who don't use trait objects directly.

use crate::api::tensor::ops::TensorOps;
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
pub fn apply_runtime_config(config: &crate::core::runtime::runtime_config::RuntimeConfig) -> TensorResult<()> {
    crate::core::runtime::runtime_config::apply_runtime_config(config)
}

// ==================== TensorBuilder factory ====================

/// Create a new [`TensorBuilder`](crate::api::tensor_builder_def::TensorBuilder) for
/// constructing tensors with explicit shape, dtype, and device.
pub fn create_tensor_builder() -> crate::api::tensor_builder_def::TensorBuilder {
    crate::api::tensor_builder_def::TensorBuilder::new()
}

// ==================== QuantConfig wrappers ====================

pub use crate::api::quant_config_def::QuantConfig;

/// Create a QuantConfig with all layers set to Q8_0.
pub fn quant_config_q8_all() -> QuantConfig {
    QuantConfig::q8_all()
}

/// Create a QuantConfig with no quantization.
pub fn quant_config_none() -> QuantConfig {
    QuantConfig::none()
}

/// Load a QuantConfig from a TOML file, falling back to defaults.
pub fn quant_config_from_toml_file(path: &std::path::Path) -> QuantConfig {
    QuantConfig::from_toml_file(path)
}

/// Parse a QuantConfig from an in-memory TOML string. Expects a
/// `[quantization]` table; unknown sections are ignored.
pub fn quant_config_from_toml_str(toml_str: &str) -> QuantConfig {
    QuantConfig::from_toml_str(toml_str)
}

/// Get the attention quantization target.
pub fn quant_config_attention(c: &QuantConfig) -> crate::api::quant::target::QuantTarget {
    c.attention()
}

/// Get the feed-forward quantization target.
pub fn quant_config_feed_forward(c: &QuantConfig) -> crate::api::quant::target::QuantTarget {
    c.feed_forward()
}

/// Get the output quantization target.
pub fn quant_config_output(c: &QuantConfig) -> crate::api::quant::target::QuantTarget {
    c.output()
}

/// Get the MoE quantization target.
pub fn quant_config_moe(c: &QuantConfig) -> crate::api::quant::target::QuantTarget {
    c.moe()
}

/// Get the gate quantization target.
pub fn quant_config_gate(c: &QuantConfig) -> crate::api::quant::target::QuantTarget {
    c.gate()
}

/// Get the minimum dimension for quantization.
pub fn quant_config_min_dim(c: &QuantConfig) -> usize {
    c.min_dim()
}

/// Set the minimum dimension for quantization.
pub fn quant_config_set_min_dim(c: &mut QuantConfig, min_dim: usize) {
    c.set_min_dim(min_dim);
}

/// Warm up the rayon thread pool by forcing all threads to wake and do work.
pub fn warmup_thread_pool() {
    crate::core::runtime::runtime_config::warmup_thread_pool();
}

/// Detect available SIMD instruction sets.
pub fn detect_simd() -> &'static str {
    crate::core::runtime::runtime_config::detect_simd()
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
        let config = crate::core::runtime::runtime_config::RuntimeConfig::default();
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

    /// @covers: create_tensor_builder
    #[test]
    fn test_create_tensor_builder_produces_valid_tensor() {
        let t = create_tensor_builder()
            .shape(vec![2, 3])
            .zeros()
            .unwrap();
        assert_eq!(t.shape(), &[2, 3]);
    }

    /// @covers: quant_config_q8_all
    #[test]
    fn test_quant_config_q8_all_returns_q8_attention() {
        let c = quant_config_q8_all();
        assert_eq!(quant_config_attention(&c), crate::api::quant::target::QuantTarget::Q8_0);
    }

    /// @covers: quant_config_none
    #[test]
    fn test_quant_config_none_returns_none_attention() {
        let c = quant_config_none();
        assert_eq!(quant_config_attention(&c), crate::api::quant::target::QuantTarget::None);
    }

    /// @covers: quant_config_set_min_dim
    #[test]
    fn test_quant_config_set_min_dim_roundtrip() {
        let mut c = quant_config_q8_all();
        quant_config_set_min_dim(&mut c, 1024);
        assert_eq!(quant_config_min_dim(&c), 1024);
    }

    /// @covers: quant_config_from_toml_file
    #[test]
    fn test_quant_config_from_toml_file_missing_returns_default() {
        let c = quant_config_from_toml_file(std::path::Path::new("/nonexistent.toml"));
        assert_eq!(quant_config_attention(&c), crate::api::quant::target::QuantTarget::Q8_0);
    }

    /// @covers: quant_config_feed_forward
    #[test]
    fn test_quant_config_feed_forward_wrapper() {
        let c = quant_config_q8_all();
        assert_eq!(quant_config_feed_forward(&c), crate::api::quant::target::QuantTarget::Q8_0);
    }

    /// @covers: quant_config_output
    #[test]
    fn test_quant_config_output_wrapper() {
        let c = quant_config_q8_all();
        assert_eq!(quant_config_output(&c), crate::api::quant::target::QuantTarget::Q8_0);
    }

    /// @covers: quant_config_moe
    #[test]
    fn test_quant_config_moe_wrapper() {
        let c = quant_config_q8_all();
        assert_eq!(quant_config_moe(&c), crate::api::quant::target::QuantTarget::Q8_0);
    }

    /// @covers: quant_config_gate
    #[test]
    fn test_quant_config_gate_wrapper() {
        let c = quant_config_q8_all();
        assert_eq!(quant_config_gate(&c), crate::api::quant::target::QuantTarget::Q8_0);
    }

    /// @covers: quant_config_min_dim
    #[test]
    fn test_quant_config_min_dim_wrapper() {
        let c = quant_config_q8_all();
        assert_eq!(quant_config_min_dim(&c), 0);
    }

    /// @covers: quant_config_attention
    #[test]
    fn test_quant_config_attention_returns_target() {
        let c = quant_config_q8_all();
        assert_eq!(quant_config_attention(&c), crate::api::quant::target::QuantTarget::Q8_0);
    }
}
