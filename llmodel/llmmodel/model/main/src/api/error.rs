use thiserror::Error;

pub type ModelResult<T> = Result<T, ModelError>;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Tensor error: {0}")]
    TensorError(#[from] swe_ml_tensor::TensorError),

    #[error("Neural network error: {0}")]
    NnError(#[from] swe_llmmodel_layers::NnError),

    #[error("Weight error: {0}")]
    WeightError(#[from] swe_llmmodel_weights::WeightError),

    #[error("Model error: {0}")]
    Model(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
