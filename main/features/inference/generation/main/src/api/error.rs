use thiserror::Error;

pub type GenerationResult<T> = Result<T, GenerationError>;

#[derive(Error, Debug)]
pub enum GenerationError {
    #[error("Model error: {0}")]
    ModelError(#[from] rustml_model::ModelError),

    #[error("Tensor error: {0}")]
    TensorError(#[from] swe_ml_tensor::TensorError),

    #[error("Neural network error: {0}")]
    NnError(#[from] rustml_inference_layers::NnError),

    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] rustml_tokenizer::TokenizerError),

    #[error("Generation error: {0}")]
    Generation(String),
}
