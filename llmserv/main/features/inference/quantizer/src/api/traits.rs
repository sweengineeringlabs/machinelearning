use swe_llmmodel_model::LlmModel;

/// Contract for model weight quantization strategies.
///
/// Implementors compress model weights after construction to reduce
/// memory bandwidth during inference. Each strategy can target different
/// layer types (attention, FFN, output head) with different quantization
/// levels (Q4_0, Q4_1, Q8_0, F16).
pub trait Quantizer: Send + Sync {
    /// Quantize the model's weights in place.
    ///
    /// Returns the number of layers successfully quantized.
    /// Idempotent: already-quantized weights are skipped.
    fn quantize(&self, model: &mut LlmModel) -> Result<usize, String>;

    /// Returns a human-readable description of the strategy.
    fn describe(&self) -> &str;
}

/// Contract for weight fusion strategies.
///
/// Implementors fuse separate weight matrices into combined tensors
/// to reduce matmul dispatch overhead during decoding.
pub trait Fuser: Send + Sync {
    /// Fuse weights in the model in place.
    ///
    /// Returns the number of fusions performed.
    fn fuse(&self, model: &mut LlmModel) -> usize;

    /// Returns a human-readable description of the fusion.
    fn describe(&self) -> &str;
}
