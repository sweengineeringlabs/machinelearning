use swe_llmmodel_model::{LlmModel, OptProfile};
use swe_llmmodel_tokenizer::Tokenizer;

/// A fully-loaded model ready for inference. Callers wrap this into
/// their own runtime type (e.g. daemon's `DefaultModel`).
pub struct LoadedModel {
    pub model: LlmModel,
    pub tokenizer: Box<dyn Tokenizer + Send + Sync>,
    pub model_id: String,
    pub chat_template: Option<String>,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub profile: OptProfile,
}
