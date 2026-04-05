//! Embedding layer implementation

use std::time::Instant;
use crate::api::error::NnResult;
use rustml_core::Tensor;

/// Embedding layer that maps token indices to dense vectors
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Embedding weight matrix [num_embeddings, embedding_dim]
    pub weight: Tensor,
    /// Number of embeddings (vocabulary size)
    pub num_embeddings: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl Embedding {
    /// Create a new embedding layer with random initialization
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        // Standard normal initialization scaled by 0.02 (GPT-2 style)
        let weight = Tensor::randn(vec![num_embeddings, embedding_dim]).mul_scalar(0.02);

        Self {
            weight,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Create an embedding layer from existing weights
    pub fn from_weights(weight: Tensor) -> NnResult<Self> {
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(crate::api::error::NnError::InvalidConfig(
                "Embedding weight must be 2D".into(),
            ));
        }

        Ok(Self {
            num_embeddings: shape[0],
            embedding_dim: shape[1],
            weight,
        })
    }

    /// Forward pass: lookup embeddings for input indices
    ///
    /// Input shape: [...] (tensor of integer indices)
    /// Output shape: [..., embedding_dim]
    pub fn forward(&self, indices: &Tensor) -> NnResult<Tensor> {
        let _t = if log::log_enabled!(log::Level::Trace) { Some(Instant::now()) } else { None };

        let input_shape = indices.shape();
        let numel = indices.numel();

        // Gather embeddings
        let mut output_data = Vec::with_capacity(numel * self.embedding_dim);

        for idx_f32 in indices.iter() {
            let idx = idx_f32 as usize;
            if idx >= self.num_embeddings {
                return Err(crate::api::error::NnError::InvalidConfig(format!(
                    "Index {} out of bounds for embedding with {} entries",
                    idx, self.num_embeddings
                )));
            }

            // Get embedding vector for this index
            let embedding = self.weight.select(0, idx)?;
            output_data.extend(embedding.iter());
        }

        // Construct output shape: input_shape + [embedding_dim]
        let mut output_shape = input_shape.to_vec();
        output_shape.push(self.embedding_dim);

        let result = Tensor::from_vec(output_data, output_shape)?;
        if let Some(t) = _t {
            log::trace!("[perf] embedding::forward {:?}->{:?} {:.3}ms",
                input_shape, result.shape(), t.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(result)
    }
}

use crate::core::linear::Linear;

/// Per-Layer Embedding (PLE) for Gemma 4.
///
/// Wraps a lookup table and a projection layer that maps the lookup
/// output to the model's hidden dimension.
#[derive(Debug, Clone)]
pub struct PerLayerEmbedding {
    pub embedding: Embedding,
    pub projection: Linear,
}

impl PerLayerEmbedding {
    pub fn new(vocab_size: usize, ple_dim: usize, model_dim: usize) -> Self {
        Self {
            embedding: Embedding::new(vocab_size, ple_dim),
            projection: Linear::new(ple_dim, model_dim),
        }
    }

    pub fn from_weights(embedding_weight: Tensor, projection_weight: Tensor) -> NnResult<Self> {
        Ok(Self {
            embedding: Embedding::from_weights(embedding_weight)?,
            projection: Linear::from_weights(projection_weight, None)?,
        })
    }

    /// Forward pass: lookup then project.
    ///
    /// Input: [B, S] (indices)
    /// Output: [B, S, model_dim]
    pub fn forward(&self, indices: &Tensor) -> NnResult<Tensor> {
        let lookup = self.embedding.forward(indices)?;
        self.projection.forward(&lookup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ple_lookup() {
        let ple = PerLayerEmbedding::new(100, 16, 64);
        let indices = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let output = ple.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[1, 3, 64]);
    }

    #[test]
    fn test_embedding_forward() {
        let embedding = Embedding::new(100, 32);
        let indices = Tensor::from_vec(vec![0.0, 5.0, 10.0, 50.0], vec![2, 2]).unwrap();
        let output = embedding.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[2, 2, 32]);
    }

    #[test]
    fn test_embedding_1d() {
        let embedding = Embedding::new(100, 64);
        let indices = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let output = embedding.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[3, 64]);
    }
}
