//! Builder for constructing QuantConfig incrementally.

use crate::api::quant::target::QuantTarget;
use crate::api::quant_config_def::QuantConfig;

/// Builder for constructing a [`QuantConfig`] incrementally.
pub struct QuantConfigBuilder {
    inner: QuantConfig,
}

impl QuantConfigBuilder {
    /// Start from the default config (Q8_0 for all layers).
    pub(crate) fn new() -> Self {
        Self {
            inner: QuantConfig {
                attention: QuantTarget::Q8_0,
                feed_forward: QuantTarget::Q8_0,
                output: QuantTarget::Q8_0,
                moe: QuantTarget::Q8_0,
                gate: QuantTarget::Q8_0,
                min_dim: 0,
            },
        }
    }

    /// Set the attention target.
    pub(crate) fn attention(mut self, target: QuantTarget) -> Self {
        self.inner.attention = target;
        self
    }

    /// Set the feed-forward target.
    pub(crate) fn feed_forward(mut self, target: QuantTarget) -> Self {
        self.inner.feed_forward = target;
        self
    }

    /// Set the output target.
    pub(crate) fn output(mut self, target: QuantTarget) -> Self {
        self.inner.output = target;
        self
    }

    /// Set the MoE target.
    pub(crate) fn moe(mut self, target: QuantTarget) -> Self {
        self.inner.moe = target;
        self
    }

    /// Set the gate target.
    pub(crate) fn gate(mut self, target: QuantTarget) -> Self {
        self.inner.gate = target;
        self
    }

    /// Set the minimum dimension.
    pub(crate) fn min_dim(mut self, dim: usize) -> Self {
        self.inner.min_dim = dim;
        self
    }

    /// Build the final [`QuantConfig`].
    pub(crate) fn build(self) -> QuantConfig {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default_produces_q8_all() {
        let c = QuantConfigBuilder::new().build();
        assert_eq!(c.attention(), QuantTarget::Q8_0);
    }

    #[test]
    fn test_builder_attention_override() {
        let c = QuantConfigBuilder::new().attention(QuantTarget::None).build();
        assert_eq!(c.attention(), QuantTarget::None);
    }
}
