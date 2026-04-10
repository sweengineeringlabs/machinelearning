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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_target_clone_preserves_equality() {
        let t = QuantTarget::Q8_0;
        assert_eq!(t, t.clone());
    }

    #[test]
    fn test_quant_target_debug_shows_variant() {
        assert_eq!(format!("{:?}", QuantTarget::None), "None");
        assert_eq!(format!("{:?}", QuantTarget::F16), "F16");
        assert_eq!(format!("{:?}", QuantTarget::Q8_0), "Q8_0");
    }

    #[test]
    fn test_quant_target_deserializes_snake_case_via_toml() {
        #[derive(serde::Deserialize)]
        struct QuantTargetToml { target: QuantTarget }
        let w: QuantTargetToml = toml::from_str("target = \"q4_0\"").unwrap();
        assert_eq!(w.target, QuantTarget::Q4_0);
    }

    #[test]
    fn test_quant_target_deserializes_none_via_toml() {
        #[derive(serde::Deserialize)]
        struct QuantTargetToml { target: QuantTarget }
        let w: QuantTargetToml = toml::from_str("target = \"none\"").unwrap();
        assert_eq!(w.target, QuantTarget::None);
    }
}
