use thiserror::Error;

pub type LoadConfigResult<T> = Result<T, LoadConfigError>;

#[derive(Error, Debug)]
pub enum LoadConfigError {
    #[error("Failed to parse bundled default config: {0}")]
    ParseDefault(String),

    #[error("Failed to read {path}: {source}")]
    ReadFile {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse {path}: {source}")]
    ParseFile {
        path: String,
        #[source]
        source: toml::de::Error,
    },

    #[error("Failed to serialize merged config: {0}")]
    Serialize(String),

    #[error("Merged config does not match schema: {0}")]
    Schema(String),
}
