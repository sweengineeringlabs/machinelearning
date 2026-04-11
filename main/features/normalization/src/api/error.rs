use swe_ml_tensor::TensorError;
use thiserror::Error;

pub type NnLayerResult<T> = Result<T, NnLayerError>;

#[derive(Error, Debug)]
pub enum NnLayerError {
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}
