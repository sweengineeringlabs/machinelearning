use crate::api::error::{ModelError, ModelResult};
use swe_ml_tensor::Tensor;

/// Token embedding — a weight matrix that maps token IDs to dense vectors.
///
/// This is the model's internal embedding lookup, not a standalone abstraction.
/// It stores a weight tensor of shape `[vocab_size, d_model]` and indexes
/// into it by row to produce token vectors.
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    weight: Tensor,
    vocab_size: usize,
    dim: usize,
}

impl TokenEmbedding {
    /// Create with random initialization (scaled by 0.02).
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        let weight = Tensor::randn(vec![vocab_size, dim]).mul_scalar(0.02);
        Self { weight, vocab_size, dim }
    }

    /// Create from a loaded weight tensor of shape `[vocab_size, dim]`.
    pub fn from_weights(weight: Tensor) -> ModelResult<Self> {
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(ModelError::Model("Embedding weight must be 2D".into()));
        }
        Ok(Self {
            vocab_size: shape[0],
            dim: shape[1],
            weight,
        })
    }

    /// Look up token vectors for the given indices.
    ///
    /// `indices` contains token IDs as f32 values.
    /// Returns a tensor with shape `[...input_shape, dim]`.
    pub fn forward(&self, indices: &Tensor) -> ModelResult<Tensor> {
        let input_shape = indices.shape();
        let numel = indices.numel();

        let mut output_data = Vec::with_capacity(numel * self.dim);

        for idx_f32 in indices.iter() {
            let idx = idx_f32 as usize;
            if idx >= self.vocab_size {
                return Err(ModelError::Model(format!(
                    "Token index {} out of bounds for vocab size {}",
                    idx, self.vocab_size
                )));
            }
            let row = self.weight.select(0, idx)?;
            output_data.extend(row.iter());
        }

        let mut output_shape = input_shape.to_vec();
        output_shape.push(self.dim);

        Ok(Tensor::from_vec(output_data, output_shape)?)
    }

    /// Returns a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_produces_correct_shape() {
        let emb = TokenEmbedding::new(100, 32);
        let ids = Tensor::from_vec(vec![0.0, 5.0, 10.0], vec![1, 3]).unwrap();
        let out = emb.forward(&ids).unwrap();
        assert_eq!(out.shape(), &[1, 3, 32]);
    }

    #[test]
    fn test_forward_out_of_bounds_rejected() {
        let emb = TokenEmbedding::new(10, 8);
        let ids = Tensor::from_vec(vec![15.0], vec![1]).unwrap();
        assert!(emb.forward(&ids).is_err());
    }

    #[test]
    fn test_from_weights_validates_shape() {
        let w = Tensor::randn(vec![5]);
        assert!(TokenEmbedding::from_weights(w).is_err());
    }

    #[test]
    fn test_same_index_same_vector() {
        let emb = TokenEmbedding::new(100, 8);
        let ids = Tensor::from_vec(vec![7.0, 7.0], vec![2]).unwrap();
        let out = emb.forward(&ids).unwrap();
        let data = out.as_slice_f32().unwrap();
        for i in 0..8 {
            assert_eq!(data[i], data[8 + i]);
        }
    }

    #[test]
    fn test_weight_tying() {
        let emb = TokenEmbedding::new(50, 16);
        assert_eq!(emb.weight().shape(), &[50, 16]);
        assert_eq!(emb.vocab_size(), 50);
        assert_eq!(emb.embedding_dim(), 16);
    }
}
