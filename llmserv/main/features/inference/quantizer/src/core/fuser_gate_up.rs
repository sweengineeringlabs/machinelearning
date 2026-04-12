use crate::api::traits::Fuser;
use rustml_model::LlmModel;

/// Fuses gate and up projection weights for SwiGLU/GeGLU FFNs.
///
/// Reduces 2 matmul dispatches to 1 during autoregressive decoding.
/// Only applies to Q8_0 layers without bias.
pub struct GateUpFuser;

impl Fuser for GateUpFuser {
    fn fuse(&self, model: &mut LlmModel) -> usize {
        model.fuse_gate_up_weights()
    }

    fn describe(&self) -> &str {
        "Fused gate+up projection (2 matmuls → 1)"
    }
}
