use crate::api::error::EmbeddingResult;
use swe_ml_tensor::Tensor;

/// Embedding layer contract.
///
/// Maps discrete integer indices to dense vectors by looking up
/// rows in a weight matrix.
pub trait Embed {
    /// Look up embeddings for the given indices.
    ///
    /// `indices` contains integer values cast as f32.
    /// Returns a tensor with an additional trailing dimension for the embedding.
    fn forward(&self, indices: &Tensor) -> EmbeddingResult<Tensor>;

    /// Returns the number of embeddings (vocabulary size).
    fn num_embeddings(&self) -> usize;

    /// Returns the embedding dimension.
    fn embedding_dim(&self) -> usize;
}
