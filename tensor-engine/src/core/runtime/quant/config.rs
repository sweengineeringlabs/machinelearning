use super::strategy::QuantStrategy;
use super::target::QuantTarget;
use crate::api::quant_ops::QuantOps;

/// Opaque quantization configuration handle.
///
/// Wraps an internal `QuantStrategy` and exposes accessors
/// without leaking the concrete type to consumers.
pub struct QuantConfig(pub(crate) QuantStrategy);

impl QuantConfig {
    /// Create with all layers set to Q8_0.
    pub fn q8_all() -> Self { Self(QuantStrategy::q8_all()) }
    /// Create with no quantization.
    pub fn none() -> Self { Self(QuantStrategy::none()) }
    /// Load from a TOML file, falling back to defaults.
    pub fn from_toml_file(path: &std::path::Path) -> Self { Self(QuantStrategy::from_toml_file(path)) }
    /// Attention target.
    pub fn attention(&self) -> QuantTarget { self.0.attention() }
    /// Feed-forward target.
    pub fn feed_forward(&self) -> QuantTarget { self.0.feed_forward() }
    /// Output target.
    pub fn output(&self) -> QuantTarget { self.0.output() }
    /// MoE target.
    pub fn moe(&self) -> QuantTarget { self.0.moe() }
    /// Gate target.
    pub fn gate(&self) -> QuantTarget { self.0.gate() }
    /// Minimum dimension.
    pub fn min_dim(&self) -> usize { self.0.min_dim() }
    /// Set minimum dimension.
    pub fn set_min_dim(&mut self, min_dim: usize) { self.0.min_dim = min_dim; }
}
