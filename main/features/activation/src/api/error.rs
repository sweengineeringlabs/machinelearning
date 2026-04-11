use swe_ml_tensor::TensorError;
use thiserror::Error;

pub type ActivationResult<T> = Result<T, ActivationError>;

#[derive(Error, Debug)]
pub enum ActivationError {
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
}
