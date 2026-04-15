//! Quantization format identifier — selects which encoding `Quantizer`
//! implementations apply.
//!
//! Only `Int8` is currently implemented end-to-end (block-symmetric,
//! per-block f32 scale). Additional variants will be added as their
//! implementations land with passing round-trip tests; until then they
//! are intentionally absent so a `QuantFormat` value is always backed
//! by real code.

use serde::{Deserialize, Serialize};

/// Supported quantization formats.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantFormat {
    /// 8-bit signed integer, block-symmetric (per-block f32 scale, zero point = 0).
    /// Storage: i8 reinterpreted as u8 byte stream + parallel f32 scales tensor.
    Int8,
}

impl QuantFormat {
    /// Bytes per quantized element in the packed payload.
    /// Int8 → 1 byte per weight.
    pub fn bytes_per_element(self) -> usize {
        match self {
            QuantFormat::Int8 => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_reports_one_byte_per_element() {
        assert_eq!(QuantFormat::Int8.bytes_per_element(), 1);
    }

    #[test]
    fn test_format_round_trips_through_serde_json() {
        let json = serde_json::to_string(&QuantFormat::Int8).unwrap();
        let back: QuantFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(back, QuantFormat::Int8);
    }
}
