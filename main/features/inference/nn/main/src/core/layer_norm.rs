//! Layer normalization implementation — delegates math to swe-ml-nn-layer.

use crate::api::error::NnResult;
use swe_ml_nn_layer::{DefaultLayerNorm, Norm};
use swe_ml_tensor::Tensor;

/// Layer normalization
///
/// Normalizes the input across the last dimension:
/// y = (x - mean) / sqrt(var + eps) * gamma + beta
///
/// Delegates the core computation to `swe_ml_nn_layer::DefaultLayerNorm`.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    inner: DefaultLayerNorm,
    /// Normalized shape (typically the last dimension)
    pub normalized_shape: usize,
}

impl LayerNorm {
    /// Create a new layer normalization
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            inner: DefaultLayerNorm::new(normalized_shape),
            normalized_shape,
        }
    }

    /// Create layer normalization with custom epsilon
    pub fn with_eps(normalized_shape: usize, eps: f32) -> Self {
        Self {
            inner: DefaultLayerNorm::with_eps(normalized_shape, eps),
            normalized_shape,
        }
    }

    /// Create from existing weights
    pub fn from_weights(weight: Tensor, bias: Tensor, eps: f32) -> NnResult<Self> {
        let normalized_shape = weight.shape()[0];
        let inner = DefaultLayerNorm::from_weights(weight, bias, eps)
            .map_err(|e| crate::api::error::NnError::InvalidConfig(e.to_string()))?;
        Ok(Self { inner, normalized_shape })
    }

    /// Returns a reference to the weight (gamma) tensor.
    pub fn weight(&self) -> &Tensor {
        self.inner.gamma()
    }

    /// Returns a reference to the bias (beta) tensor.
    pub fn bias(&self) -> &Tensor {
        self.inner.beta()
    }

    /// Forward pass
    ///
    /// Input shape: [..., normalized_shape]
    /// Output shape: [..., normalized_shape]
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        Ok(self.inner.forward(x)
            .map_err(|e| crate::api::error::NnError::InvalidConfig(e.to_string()))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(64);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_layer_norm_normalization() {
        let ln = LayerNorm::new(4);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let y = ln.forward(&x).unwrap();

        // After normalization, mean should be ~0 and std should be ~1
        let mean = y.mean(-1).unwrap();
        let var = y.var(-1).unwrap();

        assert!(mean.get(&[0]).unwrap().abs() < 1e-5);
        assert!((var.get(&[0]).unwrap() - 1.0).abs() < 0.1);
    }
}
