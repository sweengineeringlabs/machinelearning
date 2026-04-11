use crate::api::error::ActivationResult;
use swe_ml_tensor::Tensor;

/// Activation function contract.
///
/// Implementors apply a pointwise nonlinearity to the input tensor.
pub trait Activation {
    /// Apply the activation function.
    fn forward(&self, input: &Tensor) -> ActivationResult<Tensor>;
}
