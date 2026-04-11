use rustml_model::LlmModel;
use rustml_tokenizer::Tokenizer;

/// Loaded embedding model state.
pub struct EmbeddingState {
    pub model: LlmModel,
    pub tokenizer: Box<dyn Tokenizer + Send + Sync>,
    pub model_id: String,
}
