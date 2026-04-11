use std::collections::HashMap;
use rustml_model::{LlmModel, ModelBuilder, ModelConfig, ModelResult, WeightMap};
use swe_ml_tensor::Tensor;

pub struct FalconBuilder;

impl ModelBuilder for FalconBuilder {
    fn remap_weights(&self, weights: HashMap<String, Tensor>, config: &ModelConfig) -> HashMap<String, Tensor> {
        WeightMap::falcon(config.n_layers).remap(weights)
    }
    fn build(&self, config: &ModelConfig, weights: HashMap<String, Tensor>) -> ModelResult<LlmModel> {
        LlmModel::from_pretrained_falcon(config, weights)
    }
}
