use crate::api::error::ModelResult;
use crate::api::types::ModelConfig;
use crate::core::model::LlmModel;
use swe_ml_tensor::Tensor;
use std::collections::HashMap;

/// Contract for architecture-specific model builders.
///
/// Each architecture (GPT-2, Llama, Gemma, etc.) implements this trait
/// to assemble an `LlmModel` from raw weight tensors. The consumer
/// never matches on architecture names — it passes config to the
/// registry and gets a model back.
pub trait ModelBuilder: Send + Sync {
    /// Remap vendor weight names to internal names.
    ///
    /// HuggingFace and GGUF use different naming conventions per
    /// architecture. This method normalizes them before `build()`.
    fn remap_weights(
        &self,
        weights: HashMap<String, Tensor>,
        config: &ModelConfig,
    ) -> HashMap<String, Tensor>;

    /// Build an LlmModel from config and pre-remapped weights.
    fn build(
        &self,
        config: &ModelConfig,
        weights: HashMap<String, Tensor>,
    ) -> ModelResult<LlmModel>;
}
