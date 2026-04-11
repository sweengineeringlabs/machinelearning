use swe_ml_tensor::TensorError;
use thiserror::Error;

pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),

    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}
