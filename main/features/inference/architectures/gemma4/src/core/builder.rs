use std::collections::HashMap;
use rustml_model::{LlmModel, ModelBuilder, ModelConfig, ModelResult, WeightMap};
use swe_ml_tensor::Tensor;

/// Gemma 4 architecture builder.
///
/// Delegates to LlmModel::from_pretrained_gemma4() until full extraction.
/// Gemma 4 has unique features: Per-Layer Embeddings (PLE), KV sharing,
/// proportional RoPE, layer_types, double-wide MLP.
pub struct Gemma4Builder;

impl ModelBuilder for Gemma4Builder {
    fn remap_weights(&self, weights: HashMap<String, Tensor>, config: &ModelConfig) -> HashMap<String, Tensor> {
        WeightMap::gemma4(config.n_layers).remap(weights)
    }
    fn build(&self, config: &ModelConfig, weights: HashMap<String, Tensor>) -> ModelResult<LlmModel> {
        LlmModel::from_pretrained_gemma4(config, weights)
    }
}
