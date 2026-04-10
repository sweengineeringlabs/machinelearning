use super::target::QuantTarget;
use crate::api::traits::QuantOps;

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
pub(crate) struct QuantStrategy {
    /// Attention Q/K/V/O projections.
    pub(crate) attention: QuantTarget,
    /// Feed-forward up/gate/down projections.
    pub(crate) feed_forward: QuantTarget,
    /// Output (lm_head) projection.
    pub(crate) output: QuantTarget,
    /// MoE expert projections.
    pub(crate) moe: QuantTarget,
    /// PLE gate projections.
    pub(crate) gate: QuantTarget,
    /// Minimum dimension to quantize (skip smaller layers).
    pub(crate) min_dim: usize,
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

/// Conservative: keep output in F32, quantize the rest.
/// Good for models with logit softcapping (Gemma 4).
pub(crate) fn q8_preserve_output() -> QuantStrategy {
    QuantStrategy {
        output: QuantTarget::None,
        ..QuantStrategy::default()
    }
}

/// All layers in F16 — half memory of F32, no quantization noise.
pub(crate) fn f16_all() -> QuantStrategy {
    QuantStrategy {
        attention: QuantTarget::F16,
        feed_forward: QuantTarget::F16,
        output: QuantTarget::F16,
        moe: QuantTarget::F16,
        gate: QuantTarget::F16,
        min_dim: 0,
    }
}

impl QuantOps for QuantStrategy {
    fn none() -> Self {
        Self {
            attention: QuantTarget::None,
            feed_forward: QuantTarget::None,
            output: QuantTarget::None,
            moe: QuantTarget::None,
            gate: QuantTarget::None,
            min_dim: 0,
        }
    }

    fn q8_all() -> Self {
        Self::default()
    }

    fn from_toml(toml_str: &str) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        struct QuantStrategyToml {
            #[serde(default)]
            quantization: QuantStrategy,
        }
        let parsed: QuantStrategyToml = toml::from_str(toml_str)
            .map_err(|e| format!("Failed to parse quantization config: {}", e))?;
        Ok(parsed.quantization)
    }

    fn from_toml_file(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => Self::from_toml(&contents).unwrap_or_else(|e| {
                log::warn!("Invalid quantization config {}: {}. Using default.", path.display(), e);
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    fn attention(&self) -> QuantTarget { self.attention }
    fn feed_forward(&self) -> QuantTarget { self.feed_forward }
    fn output(&self) -> QuantTarget { self.output }
    fn moe(&self) -> QuantTarget { self.moe }
    fn gate(&self) -> QuantTarget { self.gate }
    fn min_dim(&self) -> usize { self.min_dim }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::traits::QuantOps;

    /// @covers: QuantStrategy (Default)
    #[test]
    fn test_quant_strategy_default_is_q8_all() {
        let s = QuantStrategy::default();
        assert_eq!(s.attention, QuantTarget::Q8_0);
        assert_eq!(s.feed_forward, QuantTarget::Q8_0);
        assert_eq!(s.output, QuantTarget::Q8_0);
    }

    /// @covers: QuantOps::none
    #[test]
    fn test_quant_strategy_none_skips_all() {
        let s = QuantStrategy::none();
        assert_eq!(s.attention, QuantTarget::None);
        assert_eq!(s.feed_forward, QuantTarget::None);
        assert_eq!(s.output, QuantTarget::None);
        assert_eq!(s.moe, QuantTarget::None);
    }

    /// @covers: QuantOps::from_toml
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

    /// @covers: QuantOps::from_toml
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

    /// @covers: QuantOps::from_toml
    #[test]
    fn test_quant_strategy_from_toml_empty_returns_defaults() {
        let s = QuantStrategy::from_toml("").unwrap();
        assert_eq!(s.attention, QuantTarget::Q8_0);
        assert_eq!(s.output, QuantTarget::Q8_0);
    }

    /// @covers: q8_preserve_output
    #[test]
    fn test_q8_preserve_output_keeps_output_in_f32() {
        let s = q8_preserve_output();
        assert_eq!(s.attention, QuantTarget::Q8_0);
        assert_eq!(s.feed_forward, QuantTarget::Q8_0);
        assert_eq!(s.output, QuantTarget::None);
    }

    /// @covers: f16_all
    #[test]
    fn test_f16_all_sets_all_to_f16() {
        let s = f16_all();
        assert_eq!(s.attention, QuantTarget::F16);
        assert_eq!(s.feed_forward, QuantTarget::F16);
        assert_eq!(s.output, QuantTarget::F16);
        assert_eq!(s.moe, QuantTarget::F16);
        assert_eq!(s.gate, QuantTarget::F16);
    }

    /// @covers: QuantOps::from_toml_file
    #[test]
    fn test_from_toml_file_missing_returns_default() {
        let s = QuantStrategy::from_toml_file(std::path::Path::new("/nonexistent.toml"));
        assert_eq!(s.attention, QuantTarget::Q8_0);
    }

    /// @covers: attention
    #[test]
    fn test_attention_returns_attention_target() {
        let s = QuantStrategy::default();
        assert_eq!(QuantOps::attention(&s), QuantTarget::Q8_0);
    }

    /// @covers: feed_forward
    #[test]
    fn test_feed_forward_returns_feed_forward_target() {
        let s = QuantStrategy::default();
        assert_eq!(QuantOps::feed_forward(&s), QuantTarget::Q8_0);
    }

    /// @covers: output
    #[test]
    fn test_output_returns_output_target() {
        let s = QuantStrategy::default();
        assert_eq!(QuantOps::output(&s), QuantTarget::Q8_0);
    }

    /// @covers: moe
    #[test]
    fn test_moe_returns_moe_target() {
        let s = QuantStrategy::default();
        assert_eq!(QuantOps::moe(&s), QuantTarget::Q8_0);
    }

    /// @covers: gate
    #[test]
    fn test_gate_returns_gate_target() {
        let s = QuantStrategy::default();
        assert_eq!(QuantOps::gate(&s), QuantTarget::Q8_0);
    }

    /// @covers: min_dim
    #[test]
    fn test_min_dim_returns_min_dimension() {
        let s = QuantStrategy::default();
        assert_eq!(QuantOps::min_dim(&s), 0);
    }
}
