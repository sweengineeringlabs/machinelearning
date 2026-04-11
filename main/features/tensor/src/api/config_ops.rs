/// Runtime configuration interface.
///
/// Provides a public method to apply runtime configuration.
pub trait ConfigOps {
    /// Apply this configuration globally.
    fn apply(&self) -> Result<(), crate::api::error::TensorError>;
}

#[cfg(test)]
mod tests {
    use crate::core::runtime::runtime_config::RuntimeConfig;
    use super::ConfigOps;

    #[test]
    fn test_config_ops_trait_apply_succeeds() {
        let config = RuntimeConfig::default();
        let result = config.apply();
        assert!(result.is_ok());
    }
}
