use crate::api::model::Model;
use crate::api::throttle::Throttle;
use rustml_inference_layers::PoolingStrategy;
use rustml_model::{LanguageModel, LlmModel, ModelResult, OptProfile};
use rustml_generation::Generator;
use rustml_tokenizer::Tokenizer;
use swe_ml_tensor::Tensor;

/// Default model instance: holds an LLM, tokenizer, and generation config.
pub struct DefaultModel {
    pub model: LlmModel,
    pub tokenizer: Box<dyn Tokenizer + Send + Sync>,
    pub model_id: String,
    pub chat_template: Option<String>,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub profile: OptProfile,
}

impl Model for DefaultModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn build_generator(&self, temperature: f32) -> Generator<'_> {
        let mut generator = Generator::new(&self.model, self.tokenizer.as_ref(), temperature);
        generator = generator.with_optimization_profile(self.profile);

        if let Some(eos) = self.eos_token_id {
            generator = generator.with_eos_token(eos);
        }
        if let Some(bos) = self.bos_token_id {
            generator = generator.with_bos_token(bos);
        }
        if self.chat_template.is_some() {
            generator = generator.with_chat_template(self.chat_template.clone());
        }

        // Auto-size context to model max
        generator = generator.with_context_len(self.model.max_sequence_length());

        generator
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    fn embed(&self, input_ids: &Tensor, strategy: PoolingStrategy) -> ModelResult<Tensor> {
        self.model.embed(input_ids, strategy)
    }
}

/// Shared application state behind `Arc` for axum handlers.
pub struct AppState {
    pub model: Box<dyn Model>,
    pub throttle: Box<dyn Throttle>,
}
