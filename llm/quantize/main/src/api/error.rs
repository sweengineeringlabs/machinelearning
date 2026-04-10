use thiserror::Error;

pub type QuantizeResult<T> = Result<T, QuantizeError>;

#[derive(Debug, Error)]
pub enum QuantizeError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("GGUF write error: {0}")]
    GgufWrite(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<swe_ml_tensor::TensorError> for QuantizeError {
    fn from(e: swe_ml_tensor::TensorError) -> Self {
        QuantizeError::Quantization(e.to_string())
    }
}

impl From<rustml_quant::QuantError> for QuantizeError {
    fn from(e: rustml_quant::QuantError) -> Self {
        QuantizeError::Quantization(e.to_string())
    }
}

impl From<rustml_gguf::GgufError> for QuantizeError {
    fn from(e: rustml_gguf::GgufError) -> Self {
        QuantizeError::GgufWrite(e.to_string())
    }
}

impl From<rustml_hub::HubError> for QuantizeError {
    fn from(e: rustml_hub::HubError) -> Self {
        QuantizeError::ModelLoad(e.to_string())
    }
}
