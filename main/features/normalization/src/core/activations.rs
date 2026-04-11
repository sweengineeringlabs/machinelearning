use crate::api::error::{NnLayerError, NnLayerResult};
use crate::api::traits::Activation;
use swe_ml_tensor::Tensor;

/// GELU activation function.
///
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// Reference: Hendrycks & Gimpel — "Gaussian Error Linear Units" (2016)
#[derive(Debug, Clone, Copy)]
pub struct Gelu;

impl Activation for Gelu {
    fn forward(&self, input: &Tensor) -> NnLayerResult<Tensor> {
        let data = input.as_slice_f32().map_err(NnLayerError::Tensor)?;
        let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();

        let output_data: Vec<f32> = data
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();

        let output = Tensor::from_vec(output_data, input.shape().to_vec())?;
        Ok(output)
    }
}

/// SiLU (Swish) activation function.
///
/// `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`
///
/// Reference: Elfwing, Uchibe, Doya — "Sigmoid-Weighted Linear Units" (2018)
#[derive(Debug, Clone, Copy)]
pub struct Silu;

impl Activation for Silu {
    fn forward(&self, input: &Tensor) -> NnLayerResult<Tensor> {
        let data = input.as_slice_f32().map_err(NnLayerError::Tensor)?;

        let output_data: Vec<f32> = data
            .iter()
            .map(|&x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            })
            .collect();

        let output = Tensor::from_vec(output_data, input.shape().to_vec())?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GELU ──

    #[test]
    fn test_gelu_forward_maps_zero_to_zero() {
        let input = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let output = Gelu.forward(&input).unwrap();
        let data = output.as_slice_f32().unwrap();
        assert!(data[0].abs() < 1e-6);
    }

    #[test]
    fn test_gelu_forward_positive_input_returns_positive() {
        let input = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let output = Gelu.forward(&input).unwrap();
        let data = output.as_slice_f32().unwrap();
        assert!(data[0] > 0.0);
    }

    #[test]
    fn test_gelu_forward_preserves_shape() {
        let input = Tensor::randn(vec![2, 3, 4]);
        let output = Gelu.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_gelu_forward_negative_input_near_zero() {
        let input = Tensor::from_vec(vec![-3.0], vec![1]).unwrap();
        let output = Gelu.forward(&input).unwrap();
        let data = output.as_slice_f32().unwrap();
        assert!(data[0].abs() < 0.01);
    }

    // ── SiLU ──

    #[test]
    fn test_silu_forward_maps_zero_to_zero() {
        let input = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let output = Silu.forward(&input).unwrap();
        let data = output.as_slice_f32().unwrap();
        assert!(data[0].abs() < 1e-6);
    }

    #[test]
    fn test_silu_forward_positive_input_returns_positive() {
        let input = Tensor::from_vec(vec![2.0], vec![1]).unwrap();
        let output = Silu.forward(&input).unwrap();
        let data = output.as_slice_f32().unwrap();
        assert!(data[0] > 0.0);
    }

    #[test]
    fn test_silu_forward_preserves_shape() {
        let input = Tensor::randn(vec![2, 3, 4]);
        let output = Silu.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_silu_forward_large_positive_approaches_identity() {
        let input = Tensor::from_vec(vec![10.0], vec![1]).unwrap();
        let output = Silu.forward(&input).unwrap();
        let data = output.as_slice_f32().unwrap();
        // sigmoid(10) ≈ 1, so silu(10) ≈ 10
        assert!((data[0] - 10.0).abs() < 0.01);
    }
}
