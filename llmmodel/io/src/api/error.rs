use thiserror::Error;

pub type IoResult<T> = Result<T, IoError>;

#[derive(Error, Debug)]
pub enum IoError {
    #[error("Invalid header: {0}")]
    InvalidHeader(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Data corruption: {0}")]
    DataCorruption(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Tensor error: {0}")]
    Tensor(#[from] swe_ml_tensor::TensorError),
}
