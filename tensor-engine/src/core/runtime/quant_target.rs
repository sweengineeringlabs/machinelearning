/// Per-layer-type quantization target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantTarget {
    /// Keep in F32 (no quantization).
    None,
    /// Half-precision float16 — lossless for BF16 models, 2 bytes/param.
    F16,
    /// Block-quantized 8-bit (default for most layers).
    Q8_0,
    /// Block-quantized 4-bit (aggressive compression).
    Q4_0,
    /// Block-quantized 4-bit with min offset (better quality than Q4_0).
    Q4_1,
}
