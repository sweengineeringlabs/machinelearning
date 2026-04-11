use thiserror::Error;

pub type ModelResult<T> = Result<T, ModelError>;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Tensor error: {0}")]
    TensorError(#[from] swe_ml_tensor::TensorError),

    #[error("Neural network error: {0}")]
    NnError(#[from] rustml_inference_layers::NnError),

    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] swe_ml_embedding::EmbeddingError),

    #[error("Hub error: {0}")]
    HubError(#[from] rustml_hub::HubError),

    #[error("Model error: {0}")]
    Model(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
