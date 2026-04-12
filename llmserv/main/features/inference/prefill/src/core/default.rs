use crate::api::traits::Prefill;
use rustml_model::{LlmModel, ModelResult, KVCache};
use swe_ml_tensor::Tensor;

/// Batch prefill — processes the entire prompt in one forward pass.
///
/// This is the default strategy. Computes all token positions in a
/// single matmul per layer, then populates the KV cache.
pub struct BatchPrefill;

impl Prefill for BatchPrefill {
    fn prefill(
        &self,
        model: &LlmModel,
        input_ids: &Tensor,
        cache: &mut KVCache,
    ) -> ModelResult<Tensor> {
        model.forward_with_cache_pass(input_ids, cache)
    }

    fn describe(&self) -> &str {
        "Batch prefill (full prompt in one pass)"
    }
}
