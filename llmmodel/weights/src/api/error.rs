use thiserror::Error;

pub type WeightResult<T> = Result<T, WeightError>;

#[derive(Error, Debug)]
pub enum WeightError {
    #[error("Missing expected weight: {0}")]
    Missing(String),

    #[error("Invalid weight shape: {0}")]
    InvalidShape(String),

    #[error("Tensor error: {0}")]
    Tensor(#[from] swe_ml_tensor::TensorError),
}
