//! Quantized tensor — opaque container holding the packed payload, the
//! per-block scale, and enough metadata to reconstruct the original shape
//! during dequantization.

use candle_core::Tensor;

use super::format::QuantFormat;

/// A tensor that has been encoded by a `Quantizer`.
///
/// Field invariants (enforced by the producing `Quantizer` impl):
/// - `data.shape()` is one-dimensional with length
///   `original_shape.iter().product::<usize>() * format.bytes_per_element()`.
/// - `scale.shape()` is one-dimensional with length
///   `ceil(original_shape.iter().product::<usize>() / block_size)`.
/// - `block_size > 0`.
pub struct QuantizedTensor {
    /// Packed payload, dtype = U8 regardless of the source dtype.
    pub data: Tensor,
    /// Per-block scaling factor, dtype = F32.
    pub scale: Tensor,
    /// Format used to produce `data`. Required for dequantization to know
    /// how to interpret the bytes.
    pub format: QuantFormat,
    /// Number of source elements per scale entry.
    pub block_size: usize,
    /// Shape of the source tensor before flattening + packing.
    pub original_shape: Vec<usize>,
}
