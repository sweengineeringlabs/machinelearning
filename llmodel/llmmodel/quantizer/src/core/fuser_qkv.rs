use crate::api::traits::Fuser;
use swe_llmmodel_model::LlmModel;

/// Fuses separate Q, K, V projection weights into a single tensor.
///
/// Reduces 3 matmul dispatches to 1 during autoregressive decoding.
/// Only applies to Q8_0 layers without bias.
pub struct QkvFuser;

impl Fuser for QkvFuser {
    fn fuse(&self, model: &mut LlmModel) -> usize {
        model.fuse_qkv_weights()
    }

    fn describe(&self) -> &str {
        "Fused QKV projection (3 matmuls → 1)"
    }
}
