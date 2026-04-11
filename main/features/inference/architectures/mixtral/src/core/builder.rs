use std::collections::HashMap;
use rustml_model::{LlmModel, ModelBuilder, ModelConfig, ModelResult, WeightMap};
use swe_ml_tensor::Tensor;

pub struct MixtralBuilder;

impl ModelBuilder for MixtralBuilder {
    fn remap_weights(&self, weights: HashMap<String, Tensor>, config: &ModelConfig) -> HashMap<String, Tensor> {
        let n_experts = config.num_local_experts.unwrap_or(8);
        WeightMap::mixtral(config.n_layers, n_experts).remap(weights)
    }
    fn build(&self, config: &ModelConfig, weights: HashMap<String, Tensor>) -> ModelResult<LlmModel> {
        LlmModel::from_pretrained_mixtral(config, weights)
    }
}
