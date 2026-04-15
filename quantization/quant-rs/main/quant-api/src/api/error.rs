//! Error type for quantization operations.

/// Errors that can occur during quantization or dequantization.
///
/// `Tensor` is wrapped behind a string because `candle_core::Error` is
/// not `Eq` and we want `QuantError` to be cheap to compare in tests.
#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    /// The tensor backend rejected an operation (shape mismatch, dtype
    /// conversion failure, allocation failure, etc.).
    #[error("Tensor error: {0}")]
    Tensor(String),
    /// The requested format has no implementation in this build.
    #[error("Unsupported quantization format: {0}")]
    Unsupported(String),
    /// A `QuantizedTensor` failed an internal invariant check
    /// (mismatched data length / scale length / block size).
    #[error("Invalid quantized tensor: {0}")]
    Invalid(String),
}

impl From<candle_core::Error> for QuantError {
    fn from(e: candle_core::Error) -> Self {
        QuantError::Tensor(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_error_unsupported_includes_format_name() {
        let err = QuantError::Unsupported("Nf4".to_string());
        assert!(err.to_string().contains("Nf4"));
    }

    #[test]
    fn test_quant_error_invalid_includes_reason() {
        let err = QuantError::Invalid("scale length 3 != n_blocks 4".to_string());
        let s = err.to_string();
        assert!(s.contains("scale length 3"));
        assert!(s.contains("n_blocks 4"));
    }
}
