//! Quantizer trait — the contract a quantization backend implements.
//!
//! Implementations live in `quant-engine`. Consumers (the CLI, evaluation
//! pipelines, downstream tooling) program against this trait so adding a
//! new format is a single new impl, not a new dispatch site.

use candle_core::Tensor;

use super::error::QuantError;
use super::tensor::QuantizedTensor;

/// Encodes and decodes tensors in a specific quantization format.
///
/// Round-trip contract: `dequantize(quantize(t))` MUST produce a tensor
/// of the same shape and dtype as `t`, with cosine similarity to `t`
/// above the format's documented tolerance. This contract is verified by
/// content-correctness tests in the implementing crate; impls that ship
/// without such a test are not considered conforming.
pub trait Quantizer: Send + Sync {
    /// Encode a tensor. Returns the packed payload and per-block scale.
    fn quantize(&self, tensor: &Tensor) -> Result<QuantizedTensor, QuantError>;

    /// Decode a previously-quantized tensor back to F32. The result has
    /// shape equal to `quantized.original_shape`.
    fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Tensor, QuantError>;
}
