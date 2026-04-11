use std::collections::HashMap;
use rustml_model::{LlmModel, ModelBuilder, ModelConfig, ModelResult, WeightMap};
use swe_ml_tensor::Tensor;

pub struct Gemma3Builder;

impl ModelBuilder for Gemma3Builder {
    fn remap_weights(&self, weights: HashMap<String, Tensor>, config: &ModelConfig) -> HashMap<String, Tensor> {
        WeightMap::gemma3(config.n_layers).remap(weights)
    }
    fn build(&self, config: &ModelConfig, weights: HashMap<String, Tensor>) -> ModelResult<LlmModel> {
        LlmModel::from_pretrained_gemma3(config, weights)
    }
}
