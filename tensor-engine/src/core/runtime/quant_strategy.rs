use super::quant_target::QuantTarget;

/// Quantization strategy: per-layer-type policies loaded from TOML.
///
/// Each field controls how a class of weights is quantized after loading.
/// `None` = keep original dtype, `Q8_0` / `Q4_0` = quantize.
///
/// ```toml
/// [quantization]
/// attention = "q8_0"
/// feed_forward = "q8_0"
/// output = "none"
/// gate = "none"
/// min_dim = 1536
/// ```
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct QuantStrategy {
    /// Attention Q/K/V/O projections.
    pub attention: QuantTarget,
    /// Feed-forward up/gate/down projections.
    pub feed_forward: QuantTarget,
    /// Output (lm_head) projection.
    pub output: QuantTarget,
    /// MoE expert projections.
    pub moe: QuantTarget,
    /// PLE gate projections.
    pub gate: QuantTarget,
    /// Minimum dimension to quantize (skip smaller layers).
    pub min_dim: usize,
}

impl Default for QuantStrategy {
    fn default() -> Self {
        Self {
            attention: QuantTarget::Q8_0,
            feed_forward: QuantTarget::Q8_0,
            output: QuantTarget::Q8_0,
            moe: QuantTarget::Q8_0,
            gate: QuantTarget::Q8_0,
            min_dim: 0,
        }
    }
}

impl QuantStrategy {
    /// All layers in F32 — no quantization.
    pub fn none() -> Self {
        Self {
            attention: QuantTarget::None,
            feed_forward: QuantTarget::None,
            output: QuantTarget::None,
            moe: QuantTarget::None,
            gate: QuantTarget::None,
            min_dim: 0,
        }
    }

    /// Default Q8_0 for all layers (current behavior).
    pub fn q8_all() -> Self {
        Self::default()
    }

    /// Conservative: keep output in F32, quantize the rest.
    /// Good for models with logit softcapping (Gemma 4).
    pub fn q8_preserve_output() -> Self {
        Self {
            output: QuantTarget::None,
            ..Self::default()
        }
    }

    /// All layers in F16 — half memory of F32, no quantization noise.
    pub fn f16_all() -> Self {
        Self {
            attention: QuantTarget::F16,
            feed_forward: QuantTarget::F16,
            output: QuantTarget::F16,
            moe: QuantTarget::F16,
            gate: QuantTarget::F16,
            min_dim: 0,
        }
    }

    /// Load from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        struct Wrapper {
            #[serde(default)]
            quantization: QuantStrategy,
        }
        let wrapper: Wrapper = toml::from_str(toml_str)
            .map_err(|e| format!("Failed to parse quantization config: {}", e))?;
        Ok(wrapper.quantization)
    }

    /// Load from a TOML file path. Falls back to default if file doesn't exist.
    pub fn from_toml_file(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => Self::from_toml(&contents).unwrap_or_else(|e| {
                log::warn!("Invalid quantization config {}: {}. Using default.", path.display(), e);
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_strategy_default_is_q8_all() {
        let s = QuantStrategy::default();
        assert_eq!(s.attention, QuantTarget::Q8_0);
        assert_eq!(s.feed_forward, QuantTarget::Q8_0);
        assert_eq!(s.output, QuantTarget::Q8_0);
    }

    #[test]
    fn test_quant_strategy_none_skips_all() {
        let s = QuantStrategy::none();
        assert_eq!(s.attention, QuantTarget::None);
        assert_eq!(s.feed_forward, QuantTarget::None);
        assert_eq!(s.output, QuantTarget::None);
        assert_eq!(s.moe, QuantTarget::None);
    }

    #[test]
    fn test_quant_strategy_from_toml_parses_mixed_targets() {
        let toml = r#"
[quantization]
attention = "q8_0"
feed_forward = "q8_0"
output = "none"
gate = "none"
min_dim = 1536
"#;
        let s = QuantStrategy::from_toml(toml).unwrap();
        assert_eq!(s.attention, QuantTarget::Q8_0);
        assert_eq!(s.output, QuantTarget::None);
        assert_eq!(s.gate, QuantTarget::None);
        assert_eq!(s.min_dim, 1536);
    }

    #[test]
    fn test_quant_strategy_from_toml_partial_uses_defaults() {
        let toml = r#"
[quantization]
output = "none"
"#;
        let s = QuantStrategy::from_toml(toml).unwrap();
        assert_eq!(s.attention, QuantTarget::Q8_0);
        assert_eq!(s.output, QuantTarget::None);
    }

    #[test]
    fn test_quant_strategy_from_toml_empty_returns_defaults() {
        let s = QuantStrategy::from_toml("").unwrap();
        assert_eq!(s.attention, QuantTarget::Q8_0);
        assert_eq!(s.output, QuantTarget::Q8_0);
    }
}
