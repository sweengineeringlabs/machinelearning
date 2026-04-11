use crate::api::error::NnLayerResult;
use swe_ml_tensor::Tensor;

/// Normalization layer contract.
///
/// Implementors provide a forward pass that normalizes the input tensor
/// across its last dimension using learned parameters.
pub trait Norm {
    /// Normalize the input tensor.
    fn forward(&self, input: &Tensor) -> NnLayerResult<Tensor>;

    /// Normalize and also return the pre-affine normalized values.
    ///
    /// Returns `(output, normalized)` where `normalized` is the values
    /// after mean/variance normalization but before the affine transform.
    /// Training uses this for the backward pass.
    fn forward_with_normalized(&self, input: &Tensor) -> NnLayerResult<(Tensor, Tensor)>;
}

/// Embedding layer contract.
///
/// Implementors map discrete integer indices to dense vectors by looking up
/// rows in a weight matrix.
pub trait Embed {
    /// Look up embeddings for the given indices.
    ///
    /// `indices` contains integer values cast as f32.
    /// Returns a tensor with an additional trailing dimension for the embedding.
    fn forward(&self, indices: &Tensor) -> NnLayerResult<Tensor>;

    /// Returns the number of embeddings (vocabulary size).
    fn num_embeddings(&self) -> usize;

    /// Returns the embedding dimension.
    fn embedding_dim(&self) -> usize;
}

/// Activation function contract.
///
/// Implementors apply a pointwise nonlinearity to the input tensor.
pub trait Activation {
    /// Apply the activation function.
    fn forward(&self, input: &Tensor) -> NnLayerResult<Tensor>;
}
