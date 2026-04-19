use std::time::Duration;

use crate::api::throttle::Throttle;
use llmbackend::Model;
use swe_llmmodel_layers::PoolingStrategy;
use swe_llmmodel_model::{LanguageModel, LlmModel, ModelResult, OptProfile};
use rustml_generation::{Generator, TextCompleter};
use swe_llmmodel_tokenizer::Tokenizer;
use swe_ml_tensor::{DType, Tensor, f32_vec_to_bytes};

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

    fn open_text_completer(&self) -> Box<dyn TextCompleter + '_> {
        // Temperature is overridden per-request via CompletionParams;
        // construct with 0.0 as an inert default. Model-level defaults
        // (profile, EOS, BOS, chat template, context length) do NOT vary
        // per request and are baked in here.
        let mut generator = Generator::new(&self.model, self.tokenizer.as_ref(), 0.0);
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

        Box::new(generator)
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    fn embed(&self, token_ids: &[u32], strategy: PoolingStrategy) -> ModelResult<Vec<f32>> {
        // The native-Rust forward pass works on `Tensor` internally.
        // Construct one here (shape [1, seq_len], F32) so the trait
        // surface doesn't have to.
        let seq_len = token_ids.len();
        let input_data: Vec<f32> = token_ids.iter().map(|&t| t as f32).collect();
        let input_tensor =
            Tensor::new(f32_vec_to_bytes(input_data), vec![1, seq_len], DType::F32);
        let out = self.model.embed(&input_tensor, strategy)?;
        Ok(out.iter().collect())
    }
}

/// Shared application state behind `Arc` for axum handlers.
pub struct AppState {
    pub model: Box<dyn Model>,
    pub throttle: Box<dyn Throttle>,
    /// Per-request wall-clock cap. `None` means no deadline (preserves
    /// the prior unbounded behavior for callers that opt out via config).
    pub request_timeout: Option<Duration>,
}
