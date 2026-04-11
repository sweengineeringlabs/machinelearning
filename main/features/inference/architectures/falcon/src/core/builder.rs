use std::collections::HashMap;
use rustml_model::{
    LlmModel, ModelBuilder, ModelConfig, ModelError, ModelResult,
};
use swe_ml_tensor::Tensor;

/// Stub builder — delegates to LlmModel::from_pretrained_* until fully extracted.
pub struct StubBuilder;

impl ModelBuilder for StubBuilder {
    fn remap_weights(
        &self,
        weights: HashMap<String, Tensor>,
        _config: &ModelConfig,
    ) -> HashMap<String, Tensor> {
        weights
    }

    fn build(
        &self,
        _config: &ModelConfig,
        _weights: HashMap<String, Tensor>,
    ) -> ModelResult<LlmModel> {
        Err(ModelError::Model("Architecture not yet extracted — use LlmModel::from_pretrained_* directly".into()))
    }
}
