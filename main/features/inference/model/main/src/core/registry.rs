use crate::api::error::{ModelError, ModelResult};
use crate::api::traits::ModelBuilder;
use crate::api::types::ModelConfig;
use crate::core::model::LlmModel;
use swe_ml_tensor::Tensor;
use std::collections::HashMap;

/// Registry of architecture-specific model builders.
///
/// Consumers register builders at startup (composition root), then
/// call `build_model()` with config and weights. The registry selects
/// the right builder based on `config.architecture` — no match
/// statements, no architecture knowledge in the consumer.
pub struct ModelBuilderRegistry {
    builders: HashMap<String, Box<dyn ModelBuilder>>,
}

impl ModelBuilderRegistry {
    pub fn new() -> Self {
        Self {
            builders: HashMap::new(),
        }
    }

    /// Register a builder for a given architecture name.
    ///
    /// Multiple names can map to the same builder (e.g., "gpt2" and "").
    pub fn register(&mut self, arch_name: &str, builder: Box<dyn ModelBuilder>) {
        self.builders.insert(arch_name.to_string(), builder);
    }

    /// Build a model from config and raw weights.
    ///
    /// Reads `config.architecture` to select the builder, remaps weights,
    /// then assembles the model. Returns an error if the architecture is
    /// not registered.
    pub fn build_model(
        &self,
        config: &ModelConfig,
        weights: HashMap<String, Tensor>,
    ) -> ModelResult<LlmModel> {
        let arch = &config.architecture;
        let builder = self.builders.get(arch).ok_or_else(|| {
            ModelError::Model(format!(
                "Unknown architecture '{}'. Registered: {:?}",
                arch,
                self.builders.keys().collect::<Vec<_>>()
            ))
        })?;
        let weights = builder.remap_weights(weights, config);
        builder.build(config, weights)
    }

    /// Returns the list of registered architecture names.
    pub fn architectures(&self) -> Vec<&str> {
        self.builders.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_registry_rejects_unknown_arch() {
        let reg = ModelBuilderRegistry::new();
        let config = ModelConfig {
            architecture: "nonexistent".to_string(),
            ..Default::default()
        };
        assert!(reg.build_model(&config, HashMap::new()).is_err());
    }

    #[test]
    fn test_architectures_lists_registered() {
        let mut reg = ModelBuilderRegistry::new();
        // Register a dummy builder
        struct DummyBuilder;
        impl ModelBuilder for DummyBuilder {
            fn remap_weights(&self, w: HashMap<String, Tensor>, _: &ModelConfig) -> HashMap<String, Tensor> { w }
            fn build(&self, _: &ModelConfig, _: HashMap<String, Tensor>) -> ModelResult<LlmModel> {
                Err(ModelError::Model("dummy".into()))
            }
        }
        reg.register("test_arch", Box::new(DummyBuilder));
        assert!(reg.architectures().contains(&"test_arch"));
    }
}
