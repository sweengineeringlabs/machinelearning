use swe_llmmodel_model::LlmModel;
use swe_llmmodel_tokenizer::Tokenizer;

/// Loaded embedding model state.
pub struct EmbeddingState {
    pub model: LlmModel,
    pub tokenizer: Box<dyn Tokenizer + Send + Sync>,
    pub model_id: String,
}
