//! Error types for crates_quant_packer.

/// Errors that can occur in crates_quant_packer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A configuration error occurred.
    #[error("Configuration error: {message}")]
    Config { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_io_display() {
        let err = Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"));
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn test_error_config_display() {
        let err = Error::Config { message: "bad".to_string() };
        assert!(err.to_string().contains("bad"));
    }
}
