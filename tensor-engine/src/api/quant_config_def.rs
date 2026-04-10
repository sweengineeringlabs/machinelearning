//! Opaque quantization configuration handle.
//!
//! A public data type defining quantization settings.
//! Implementation (loading from TOML, etc.) lives in core/runtime/quant/.

use crate::api::quant::target::QuantTarget;

/// Quantization configuration: per-layer-type quantization targets.
///
/// Each field controls how a class of weights is quantized after loading.
pub struct QuantConfig {
    /// Attention Q/K/V/O projections.
    pub(crate) attention: QuantTarget,
    /// Feed-forward up/gate/down projections.
    pub(crate) feed_forward: QuantTarget,
    /// Output (lm_head) projection.
    pub(crate) output: QuantTarget,
    /// MoE expert projections.
    pub(crate) moe: QuantTarget,
    /// PLE gate projections.
    pub(crate) gate: QuantTarget,
    /// Minimum dimension to quantize (skip smaller layers).
    pub(crate) min_dim: usize,
}

impl QuantConfig {
    /// Attention target.
    pub(crate) fn attention(&self) -> QuantTarget { self.attention }
    /// Feed-forward target.
    pub(crate) fn feed_forward(&self) -> QuantTarget { self.feed_forward }
    /// Output target.
    pub(crate) fn output(&self) -> QuantTarget { self.output }
    /// MoE target.
    pub(crate) fn moe(&self) -> QuantTarget { self.moe }
    /// Gate target.
    pub(crate) fn gate(&self) -> QuantTarget { self.gate }
    /// Minimum dimension.
    pub(crate) fn min_dim(&self) -> usize { self.min_dim }
    /// Set minimum dimension.
    pub(crate) fn set_min_dim(&mut self, min_dim: usize) { self.min_dim = min_dim; }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_q8_config() -> QuantConfig {
        QuantConfig {
            attention: QuantTarget::Q8_0,
            feed_forward: QuantTarget::Q8_0,
            output: QuantTarget::Q8_0,
            moe: QuantTarget::Q8_0,
            gate: QuantTarget::Q8_0,
            min_dim: 0,
        }
    }

    #[test]
    fn test_quant_config_attention_accessor() {
        let c = make_q8_config();
        assert_eq!(c.attention(), QuantTarget::Q8_0);
    }

    #[test]
    fn test_quant_config_feed_forward_accessor() {
        let c = make_q8_config();
        assert_eq!(c.feed_forward(), QuantTarget::Q8_0);
    }

    #[test]
    fn test_quant_config_output_accessor() {
        let c = make_q8_config();
        assert_eq!(c.output(), QuantTarget::Q8_0);
    }

    #[test]
    fn test_quant_config_moe_accessor() {
        let c = make_q8_config();
        assert_eq!(c.moe(), QuantTarget::Q8_0);
    }

    #[test]
    fn test_quant_config_gate_accessor() {
        let c = make_q8_config();
        assert_eq!(c.gate(), QuantTarget::Q8_0);
    }

    #[test]
    fn test_quant_config_min_dim_accessor() {
        let c = make_q8_config();
        assert_eq!(c.min_dim(), 0);
    }

    #[test]
    fn test_quant_config_set_min_dim_updates_value() {
        let mut c = make_q8_config();
        c.set_min_dim(2048);
        assert_eq!(c.min_dim(), 2048);
    }
}
