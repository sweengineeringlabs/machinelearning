/// Quantization strategy interface.
///
/// Defines per-layer-type quantization policies.
pub trait QuantOps {
    /// All layers in F32 — no quantization.
    fn none() -> Self where Self: Sized;
    /// Default Q8_0 for all layers.
    fn q8_all() -> Self where Self: Sized;
    /// Load from a TOML string.
    fn from_toml(toml_str: &str) -> Result<Self, String> where Self: Sized;
    /// Load from a TOML file path with fallback to defaults.
    fn from_toml_file(path: &std::path::Path) -> Self where Self: Sized;
}

#[cfg(test)]
mod tests {
    use crate::core::runtime::quant::strategy::QuantStrategy;
    use crate::core::runtime::quant::target::QuantTarget;

    #[test]
    fn test_quant_ops_none_returns_no_quantization() {
        let s = QuantStrategy::none();
        assert_eq!(s.attention, QuantTarget::None);
    }
}
