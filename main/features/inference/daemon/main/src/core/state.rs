use rustml_model::{LanguageModel, LlmModel, OptProfile};
use rustml_generation::Generator;
use rustml_tokenizer::Tokenizer;

/// Loaded model bundle: model + tokenizer + metadata needed for generation.
pub struct ModelBundle {
    pub model: LlmModel,
    pub tokenizer: Box<dyn Tokenizer + Send + Sync>,
    pub model_id: String,
    pub chat_template: Option<String>,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub profile: OptProfile,
}

impl ModelBundle {
    /// Build a `Generator` borrowing from this bundle.
    pub fn build_generator(&self, temperature: f32) -> Generator<'_> {
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
}

/// Shared application state behind `Arc` for axum handlers.
pub struct AppState {
    pub bundle: ModelBundle,
}
