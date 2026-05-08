use thiserror::Error;

pub type LoaderResult<T> = Result<T, LoaderError>;

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("Download error: {0}")]
    Download(#[from] swe_llmmodel_download::DownloadError),

    #[error("IO error: {0}")]
    Io(#[from] swe_llmmodel_io::IoError),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Config parse error: {0}")]
    Config(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Filesystem error: {0}")]
    Fs(#[from] std::io::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("GGUF error: {0}")]
    Gguf(String),
}
