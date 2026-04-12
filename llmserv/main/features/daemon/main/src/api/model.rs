use rustml_generation::Generator;
use rustml_inference_layers::PoolingStrategy;
use rustml_model::ModelResult;
use rustml_tokenizer::Tokenizer;
use swe_ml_tensor::Tensor;

/// Interface for a loaded model instance ready for inference.
pub trait Model: Send + Sync {
    /// Model identifier (e.g. HuggingFace model ID or GGUF filename).
    fn model_id(&self) -> &str;

    /// Build a generator configured for the given temperature.
    fn build_generator(&self, temperature: f32) -> Generator<'_>;

    /// Access the tokenizer.
    fn tokenizer(&self) -> &dyn Tokenizer;

    /// Compute embeddings for the given input with the specified pooling strategy.
    fn embed(&self, input_ids: &Tensor, strategy: PoolingStrategy) -> ModelResult<Tensor>;
}
