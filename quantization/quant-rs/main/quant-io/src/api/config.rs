//! Configuration type for crates_quant_io.

/// Configuration for crates_quant_io.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Enable verbose output.
    pub(crate) verbose: bool,
}

impl Config {
    /// Create a new default configuration.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Set verbose mode.
    pub(crate) fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new_returns_default() {
        let config = Config::new();
        assert!(!config.verbose);
    }

    #[test]
    fn test_config_with_verbose() {
        let config = Config::new().with_verbose(true);
        assert!(config.verbose);
    }
}
