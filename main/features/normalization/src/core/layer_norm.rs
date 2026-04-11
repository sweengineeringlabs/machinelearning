use crate::api::error::{NnLayerError, NnLayerResult};
use crate::api::traits::Norm;
use swe_ml_tensor::Tensor;

/// Default layer normalization implementation.
///
/// Normalizes the input across the last dimension:
/// `y = gamma * (x - mean) / sqrt(var + eps) + beta`
///
/// Reference: Ba, Kiros, Hinton — "Layer Normalization" (2016)
#[derive(Debug, Clone)]
pub struct DefaultLayerNorm {
    gamma: Tensor,
    beta: Tensor,
    normalized_shape: usize,
    eps: f32,
}

impl DefaultLayerNorm {
    /// Create with weights initialized to ones (gamma) and zeros (beta).
    pub fn new(normalized_shape: usize) -> Self {
        Self::with_eps(normalized_shape, 1e-5)
    }

    /// Create with custom epsilon.
    pub fn with_eps(normalized_shape: usize, eps: f32) -> Self {
        Self {
            gamma: Tensor::ones(vec![normalized_shape]),
            beta: Tensor::zeros(vec![normalized_shape]),
            normalized_shape,
            eps,
        }
    }

    /// Create from existing weight tensors.
    pub fn from_weights(gamma: Tensor, beta: Tensor, eps: f32) -> NnLayerResult<Self> {
        if gamma.shape() != beta.shape() {
            return Err(NnLayerError::ShapeMismatch(
                "Gamma and beta shapes must match".into(),
            ));
        }
        if gamma.ndim() != 1 {
            return Err(NnLayerError::InvalidConfig(
                "Gamma must be 1D".into(),
            ));
        }

        Ok(Self {
            normalized_shape: gamma.shape()[0],
            gamma,
            beta,
            eps,
        })
    }

    /// Returns a reference to the gamma (scale) parameter.
    pub fn gamma(&self) -> &Tensor {
        &self.gamma
    }

    /// Returns a reference to the beta (shift) parameter.
    pub fn beta(&self) -> &Tensor {
        &self.beta
    }

    /// Returns the normalized shape.
    pub fn normalized_shape(&self) -> usize {
        self.normalized_shape
    }

    /// Returns the epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Core normalization on raw f32 data.
    ///
    /// Returns `(output_data, normalized_data)`.
    /// `normalized_data` contains the pre-affine values needed for backward.
    fn compute(
        data: &[f32],
        gamma: &[f32],
        beta: &[f32],
        last_dim: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = data.len() / last_dim;
        let d = last_dim as f32;
        let mut normalized_data = vec![0.0f32; data.len()];
        let mut output_data = vec![0.0f32; data.len()];

        for i in 0..n {
            let start = i * last_dim;
            let row = &data[start..start + last_dim];

            let mean: f32 = row.iter().sum::<f32>() / d;
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d;
            let inv_std = 1.0 / (var + eps).sqrt();

            for j in 0..last_dim {
                let norm_val = (row[j] - mean) * inv_std;
                normalized_data[start + j] = norm_val;
                output_data[start + j] = gamma[j] * norm_val + beta[j];
            }
        }

        (output_data, normalized_data)
    }
}

impl Norm for DefaultLayerNorm {
    fn forward(&self, input: &Tensor) -> NnLayerResult<Tensor> {
        let (output, _) = self.forward_with_normalized(input)?;
        Ok(output)
    }

    fn forward_with_normalized(&self, input: &Tensor) -> NnLayerResult<(Tensor, Tensor)> {
        let shape = input.shape();
        if shape.is_empty() || shape[shape.len() - 1] != self.normalized_shape {
            return Err(NnLayerError::ShapeMismatch(format!(
                "Expected last dimension to be {}, got {:?}",
                self.normalized_shape, shape
            )));
        }

        let data = input.as_slice_f32()
            .map_err(|e| NnLayerError::Tensor(e))?;
        let gamma = self.gamma.as_slice_f32()
            .map_err(|e| NnLayerError::Tensor(e))?;
        let beta = self.beta.as_slice_f32()
            .map_err(|e| NnLayerError::Tensor(e))?;

        let (output_data, normalized_data) =
            Self::compute(data, gamma, beta, self.normalized_shape, self.eps);

        let shape_vec = shape.to_vec();
        let output = Tensor::from_vec(output_data, shape_vec.clone())?;
        let normalized = Tensor::from_vec(normalized_data, shape_vec)?;

        Ok((output, normalized))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_preserves_shape() {
        let ln = DefaultLayerNorm::new(64);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_forward_normalizes_to_zero_mean_unit_variance() {
        let ln = DefaultLayerNorm::new(4);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let y = ln.forward(&x).unwrap();

        let mean = y.mean(-1).unwrap();
        let var = y.var(-1).unwrap();

        assert!(mean.get(&[0]).unwrap().abs() < 1e-5);
        assert!((var.get(&[0]).unwrap() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_forward_with_normalized_returns_pre_affine_values() {
        let ln = DefaultLayerNorm::new(4);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let (output, normalized) = ln.forward_with_normalized(&x).unwrap();

        assert_eq!(output.shape(), &[1, 4]);
        assert_eq!(normalized.shape(), &[1, 4]);

        // With default gamma=1, beta=0, output should equal normalized
        let out_data = output.as_slice_f32().unwrap();
        let norm_data = normalized.as_slice_f32().unwrap();
        for i in 0..4 {
            assert!((out_data[i] - norm_data[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_from_weights_validates_shape_mismatch() {
        let gamma = Tensor::ones(vec![4]);
        let beta = Tensor::ones(vec![8]);
        assert!(DefaultLayerNorm::from_weights(gamma, beta, 1e-5).is_err());
    }

    #[test]
    fn test_from_weights_validates_dimensionality() {
        let gamma = Tensor::ones(vec![2, 4]);
        let beta = Tensor::ones(vec![2, 4]);
        assert!(DefaultLayerNorm::from_weights(gamma, beta, 1e-5).is_err());
    }

    #[test]
    fn test_forward_rejects_wrong_last_dim() {
        let ln = DefaultLayerNorm::new(4);
        let x = Tensor::randn(vec![2, 8]);
        assert!(ln.forward(&x).is_err());
    }

    #[test]
    fn test_with_eps_sets_epsilon() {
        let ln = DefaultLayerNorm::with_eps(4, 1e-3);
        assert!((ln.eps() - 1e-3).abs() < 1e-10);
    }
}
