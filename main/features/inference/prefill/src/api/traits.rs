use rustml_model::{LlmModel, ModelResult};
use swe_ml_tensor::Tensor;

/// Contract for prompt prefill strategies.
///
/// Prefill processes the entire input prompt before autoregressive decoding
/// begins. Different strategies trade off latency vs memory:
///
/// - **Batch**: Process all tokens in one matmul (fast, high memory)
/// - **Chunked**: Process in fixed-size chunks (bounded memory, slightly slower)
/// - **Sequential**: One token at a time (minimal memory, slowest)
pub trait Prefill: Send + Sync {
    /// Process the input prompt and return the logits for the last token.
    ///
    /// The implementation may also populate the KV cache as a side effect.
    fn prefill(
        &self,
        model: &LlmModel,
        input_ids: &Tensor,
        cache: &mut rustml_model::KVCache,
    ) -> ModelResult<Tensor>;

    /// Returns a human-readable description of the strategy.
    fn describe(&self) -> &str;
}
