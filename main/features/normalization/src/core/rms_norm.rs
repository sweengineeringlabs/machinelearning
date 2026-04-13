use crate::api::error::{NnLayerError, NnLayerResult};
use crate::api::traits::Norm;
use swe_ml_tensor::Tensor;

/// Default RMS normalization implementation.
///
/// `y = x * weight / rms(x)` where `rms(x) = sqrt(mean(x^2) + eps)`.
/// Used by Llama-family models. Unlike LayerNorm, does not subtract the mean
/// and has no bias parameter.
///
/// Supports an optional additive weight offset (Gemma uses `offset = 1.0`
/// so the effective weight becomes `w + 1.0`).
#[derive(Debug, Clone)]
pub struct DefaultRmsNorm {
    weight: Tensor,
    eps: f32,
    offset: f32,
}

impl DefaultRmsNorm {
    /// Create with weights initialized to ones.
    pub fn new(dim: usize, eps: f32) -> Self {
        Self {
            weight: Tensor::ones(vec![dim]),
            eps,
            offset: 0.0,
        }
    }

    /// Create from existing weight tensor.
    pub fn from_weight(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps, offset: 0.0 }
    }

    /// Create from existing weight tensor with an additive offset.
    ///
    /// Gemma models use `offset = 1.0` so the effective weight is `w + 1.0`.
    pub fn from_weight_with_offset(weight: Tensor, eps: f32, offset: f32) -> Self {
        Self { weight, eps, offset }
    }

    /// Returns a reference to the weight parameter.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns the epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Returns the weight offset.
    pub fn offset(&self) -> f32 {
        self.offset
    }

    /// Core RMS normalization on raw f32 data.
    fn compute(data: &[f32], weight: &[f32], last_dim: usize, eps: f32) -> Vec<f32> {
        let n = data.len() / last_dim;
        let d = last_dim as f32;
        let mut output = vec![0.0f32; data.len()];

        for i in 0..n {
            let start = i * last_dim;
            let row = &data[start..start + last_dim];

            let rms = (row.iter().map(|&x| x * x).sum::<f32>() / d + eps).sqrt();
            let inv_rms = 1.0 / rms;

            for j in 0..last_dim {
                output[start + j] = row[j] * inv_rms * weight[j];
            }
        }

        output
    }
}

impl Norm for DefaultRmsNorm {
    fn forward(&self, input: &Tensor) -> NnLayerResult<Tensor> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(NnLayerError::ShapeMismatch(
                "Input must have at least 1 dimension".into(),
            ));
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.weight.shape()[0] {
            return Err(NnLayerError::ShapeMismatch(format!(
                "Expected last dimension to be {}, got {}",
                self.weight.shape()[0], last_dim
            )));
        }

        // Critical: as_slice_f32() returns raw storage bytes regardless
        // of whether the tensor's logical layout is contiguous. If the
        // input was produced by a non-contiguous op (transpose, slice
        // view), the raw slice isn't aligned with the shape's natural
        // row-major layout — RMSNorm's per-row indexing would then
        // read scrambled data. Materialize a contiguous copy first.
        // P9 root cause: this was producing wrong outputs for
        // ffn_norm/post_ffn_norm in gemma3 layer 0 because their
        // inputs went through non-contiguous attention intermediates.
        let owned_data: Vec<f32>;
        let data: &[f32] = if input.is_contiguous() {
            input.as_slice_f32().map_err(NnLayerError::Tensor)?
        } else {
            owned_data = input.iter().collect();
            &owned_data
        };

        let effective_weight = if self.offset == 0.0 {
            self.weight.clone()
        } else {
            self.weight.add_scalar(self.offset)
        };
        // Effective weight is freshly produced by add_scalar (or a
        // clone), so it's always contiguous. Direct slice is fine.
        let weight = effective_weight.as_slice_f32().map_err(NnLayerError::Tensor)?;

        let output_data = Self::compute(data, weight, last_dim, self.eps);
        let output = Tensor::from_vec(output_data, shape.to_vec())?;
        Ok(output)
    }

    fn forward_with_normalized(&self, input: &Tensor) -> NnLayerResult<(Tensor, Tensor)> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(NnLayerError::ShapeMismatch(
                "Input must have at least 1 dimension".into(),
            ));
        }

        let last_dim = shape[shape.len() - 1];
        if last_dim != self.weight.shape()[0] {
            return Err(NnLayerError::ShapeMismatch(format!(
                "Expected last dimension to be {}, got {}",
                self.weight.shape()[0], last_dim
            )));
        }

        // Same contiguity guard as `forward` — see comment there.
        let owned_data: Vec<f32>;
        let data: &[f32] = if input.is_contiguous() {
            input.as_slice_f32().map_err(NnLayerError::Tensor)?
        } else {
            owned_data = input.iter().collect();
            &owned_data
        };

        let effective_weight = if self.offset == 0.0 {
            self.weight.clone()
        } else {
            self.weight.add_scalar(self.offset)
        };
        let weight = effective_weight.as_slice_f32().map_err(NnLayerError::Tensor)?;

        let n = data.len() / last_dim;
        let d = last_dim as f32;
        let mut normalized_data = vec![0.0f32; data.len()];
        let mut output_data = vec![0.0f32; data.len()];

        for i in 0..n {
            let start = i * last_dim;
            let row = &data[start..start + last_dim];

            let rms = (row.iter().map(|&x| x * x).sum::<f32>() / d + self.eps).sqrt();
            let inv_rms = 1.0 / rms;

            for j in 0..last_dim {
                let norm_val = row[j] * inv_rms;
                normalized_data[start + j] = norm_val;
                output_data[start + j] = norm_val * weight[j];
            }
        }

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
        let rn = DefaultRmsNorm::new(64, 1e-5);
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = rn.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_forward_with_offset_preserves_shape() {
        let rn = DefaultRmsNorm::from_weight_with_offset(
            Tensor::zeros(vec![64]), 1e-5, 1.0,
        );
        let x = Tensor::randn(vec![2, 10, 64]);
        let y = rn.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_zero_offset_matches_no_offset() {
        let weight = Tensor::randn(vec![32]);
        let standard = DefaultRmsNorm::from_weight(weight.clone(), 1e-5);
        let with_offset = DefaultRmsNorm::from_weight_with_offset(weight, 1e-5, 0.0);
        let x = Tensor::randn(vec![1, 4, 32]);

        let y_std = standard.forward(&x).unwrap();
        let y_off = with_offset.forward(&x).unwrap();

        let d_std = y_std.as_slice_f32().unwrap();
        let d_off = y_off.as_slice_f32().unwrap();
        for i in 0..d_std.len() {
            assert!(
                (d_std[i] - d_off[i]).abs() < 1e-5,
                "mismatch at index {}: {} vs {}", i, d_std[i], d_off[i]
            );
        }
    }

    #[test]
    fn test_forward_rejects_wrong_last_dim() {
        let rn = DefaultRmsNorm::new(4, 1e-5);
        let x = Tensor::randn(vec![2, 8]);
        assert!(rn.forward(&x).is_err());
    }

    #[test]
    fn test_forward_with_normalized_returns_both() {
        let rn = DefaultRmsNorm::new(8, 1e-5);
        let x = Tensor::randn(vec![2, 8]);
        let (output, normalized) = rn.forward_with_normalized(&x).unwrap();
        assert_eq!(output.shape(), &[2, 8]);
        assert_eq!(normalized.shape(), &[2, 8]);
    }
}
