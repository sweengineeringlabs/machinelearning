use crate::api::error::{EmbeddingError, EmbeddingResult};
use crate::api::traits::Embed;
use swe_ml_tensor::Tensor;

/// Default embedding layer implementation.
///
/// Maps discrete token indices to dense vectors by looking up rows
/// in a weight matrix of shape `[num_embeddings, embedding_dim]`.
#[derive(Debug, Clone)]
pub struct DefaultEmbedding {
    weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl DefaultEmbedding {
    /// Create with random initialization (scaled by 0.02, GPT-2 style).
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight = Tensor::randn(vec![num_embeddings, embedding_dim]).mul_scalar(0.02);
        Self {
            weight,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Create from an existing weight tensor of shape `[num_embeddings, embedding_dim]`.
    pub fn from_weights(weight: Tensor) -> EmbeddingResult<Self> {
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(EmbeddingError::InvalidConfig(
                "Embedding weight must be 2D".into(),
            ));
        }

        Ok(Self {
            num_embeddings: shape[0],
            embedding_dim: shape[1],
            weight,
        })
    }

    /// Returns a reference to the weight matrix.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Embed for DefaultEmbedding {
    fn forward(&self, indices: &Tensor) -> EmbeddingResult<Tensor> {
        let input_shape = indices.shape();
        let numel = indices.numel();

        let mut output_data = Vec::with_capacity(numel * self.embedding_dim);

        for idx_f32 in indices.iter() {
            let idx = idx_f32 as usize;
            if idx >= self.num_embeddings {
                return Err(EmbeddingError::IndexOutOfBounds(format!(
                    "Index {} out of bounds for embedding with {} entries",
                    idx, self.num_embeddings
                )));
            }

            let embedding = self.weight.select(0, idx)?;
            output_data.extend(embedding.iter());
        }

        let mut output_shape = input_shape.to_vec();
        output_shape.push(self.embedding_dim);

        let result = Tensor::from_vec(output_data, output_shape)?;
        Ok(result)
    }

    fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_2d_indices() {
        let emb = DefaultEmbedding::new(100, 32);
        let indices = Tensor::from_vec(vec![0.0, 5.0, 10.0, 50.0], vec![2, 2]).unwrap();
        let output = emb.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[2, 2, 32]);
    }

    #[test]
    fn test_forward_1d_indices() {
        let emb = DefaultEmbedding::new(100, 64);
        let indices = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let output = emb.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[3, 64]);
    }

    #[test]
    fn test_forward_out_of_bounds_rejected() {
        let emb = DefaultEmbedding::new(10, 8);
        let indices = Tensor::from_vec(vec![15.0], vec![1]).unwrap();
        assert!(emb.forward(&indices).is_err());
    }

    #[test]
    fn test_from_weights_validates_dimensionality() {
        let weight = Tensor::randn(vec![5]);
        assert!(DefaultEmbedding::from_weights(weight).is_err());
    }

    #[test]
    fn test_from_weights_sets_dimensions() {
        let weight = Tensor::randn(vec![50, 16]);
        let emb = DefaultEmbedding::from_weights(weight).unwrap();
        assert_eq!(emb.num_embeddings(), 50);
        assert_eq!(emb.embedding_dim(), 16);
    }

    #[test]
    fn test_same_index_returns_same_embedding() {
        let emb = DefaultEmbedding::new(100, 8);
        let idx = Tensor::from_vec(vec![7.0, 7.0], vec![2]).unwrap();
        let output = emb.forward(&idx).unwrap();
        let data = output.as_slice_f32().unwrap();
        for i in 0..8 {
            assert_eq!(data[i], data[8 + i]);
        }
    }
}
