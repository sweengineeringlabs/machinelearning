use thiserror::Error;

pub type DownloadResult<T> = Result<T, DownloadError>;

#[derive(Error, Debug)]
pub enum DownloadError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Auth error: {0}")]
    Auth(String),
}
