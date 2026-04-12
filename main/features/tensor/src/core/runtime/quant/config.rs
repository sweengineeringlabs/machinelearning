use super::strategy::QuantStrategy;
use crate::api::quant_config_def::QuantConfig;
use crate::api::traits::QuantOps;

/// Namespace marker for QuantConfig implementation methods defined in this module.
pub(crate) struct Config;

/// Extension methods on QuantConfig that require core/ access.
impl QuantConfig {
    /// Create with all layers set to Q8_0.
    pub(crate) fn q8_all() -> Self {
        let s = QuantStrategy::q8_all();
        Self::from_strategy(s)
    }

    /// Create with no quantization.
    pub(crate) fn none() -> Self {
        let s = QuantStrategy::none();
        Self::from_strategy(s)
    }

    /// Load from a TOML file, falling back to defaults.
    pub(crate) fn from_toml_file(path: &std::path::Path) -> Self {
        let s = QuantStrategy::from_toml_file(path);
        Self::from_strategy(s)
    }

    /// Parse from an in-memory TOML string. The TOML must contain a
    /// `[quantization]` table; unknown sections are ignored. On parse
    /// error, falls back to defaults.
    pub(crate) fn from_toml_str(toml_str: &str) -> Self {
        let s = QuantStrategy::from_toml(toml_str).unwrap_or_else(|e| {
            log::warn!("Invalid quantization TOML: {}. Using defaults.", e);
            QuantStrategy::default()
        });
        Self::from_strategy(s)
    }

    /// Convert a QuantStrategy into a QuantConfig.
    pub(crate) fn from_strategy(s: QuantStrategy) -> Self {
        Self {
            attention: s.attention,
            feed_forward: s.feed_forward,
            output: s.output,
            moe: s.moe,
            gate: s.gate,
            min_dim: s.min_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::quant::target::QuantTarget;

    /// @covers: QuantConfig::q8_all
    #[test]
    fn test_quant_config_q8_all_sets_attention_to_q8() {
        let c = QuantConfig::q8_all();
        assert_eq!(c.attention(), QuantTarget::Q8_0);
        assert_eq!(c.feed_forward(), QuantTarget::Q8_0);
    }

    /// @covers: QuantConfig::none
    #[test]
    fn test_quant_config_none_sets_all_to_none() {
        let c = QuantConfig::none();
        assert_eq!(c.attention(), QuantTarget::None);
        assert_eq!(c.feed_forward(), QuantTarget::None);
    }

    /// @covers: QuantConfig::from_toml_file
    #[test]
    fn test_quant_config_from_toml_file_missing_returns_default() {
        let c = QuantConfig::from_toml_file(std::path::Path::new("/nonexistent.toml"));
        assert_eq!(c.attention(), QuantTarget::Q8_0);
    }

    /// @covers: QuantConfig::from_strategy
    #[test]
    fn test_from_strategy_copies_all_fields() {
        let s = QuantStrategy::none();
        let c = QuantConfig::from_strategy(s);
        assert_eq!(c.attention(), QuantTarget::None);
        assert_eq!(c.min_dim(), 0);
    }
}
