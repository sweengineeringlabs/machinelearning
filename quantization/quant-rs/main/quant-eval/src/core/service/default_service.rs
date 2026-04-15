//! Default Service implementation.

use crate::api::config::Config;
use crate::api::error::Error;
use crate::api::traits::Service;

/// Default implementation of the Service trait.
#[derive(Debug, Default)]
pub(crate) struct DefaultService;

impl DefaultService {
    /// Create a new default service.
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Service for DefaultService {
    fn execute(&self, config: &Config) -> Result<(), Error> {
        if config.verbose {
            println!("[crates/quant-eval] executing with verbose=true");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_default_service() {
        let _service = DefaultService::new();
    }

    #[test]
    fn test_execute_succeeds_with_default_config() {
        let service = DefaultService::new();
        let config = Config::default();
        assert!(service.execute(&config).is_ok());
    }

    #[test]
    fn test_execute_succeeds_in_verbose_mode() {
        let service = DefaultService::new();
        let config = Config::default().with_verbose(true);
        assert!(service.execute(&config).is_ok());
    }
}
