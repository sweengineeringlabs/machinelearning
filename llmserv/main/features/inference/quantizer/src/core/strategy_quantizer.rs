use crate::api::traits::Quantizer;
use rustml_model::LlmModel;
use swe_ml_tensor::QuantConfig;

/// Config-driven quantizer — delegates to LlmModel::quantize_with_strategy.
///
/// Uses a QuantConfig to specify per-layer-type quantization targets
/// (Q8_0, Q4_0, Q4_1, F16) with minimum dimension thresholds.
pub struct ConfigQuantizer {
    strategy: QuantConfig,
}

impl ConfigQuantizer {
    pub fn new(strategy: QuantConfig) -> Self {
        Self { strategy }
    }

    /// Load quantization config from a TOML file.
    pub fn from_toml(path: &std::path::Path) -> Self {
        let strategy = swe_ml_tensor::quant_config_from_toml_file(path);
        Self { strategy }
    }

    /// Create with uniform Q8_0 quantization for all layers.
    pub fn q8_all() -> Self {
        Self {
            strategy: swe_ml_tensor::quant_config_q8_all(),
        }
    }
}

impl Quantizer for ConfigQuantizer {
    fn quantize(&self, model: &mut LlmModel) -> Result<usize, String> {
        model.quantize_with_strategy(&self.strategy).map_err(|e| e.to_string())
    }

    fn describe(&self) -> &str {
        "Config-driven quantization (per-layer-type targets)"
    }
}

/// Quantize only the output head (lm_head) to Q8_0.
pub struct LmHeadQuantizer;

impl Quantizer for LmHeadQuantizer {
    fn quantize(&self, model: &mut LlmModel) -> Result<usize, String> {
        model.quantize_lm_head().map_err(|e| e.to_string())?;
        Ok(1)
    }

    fn describe(&self) -> &str {
        "LM head Q8_0 quantization"
    }
}
