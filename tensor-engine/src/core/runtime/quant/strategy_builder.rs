use super::strategy::QuantStrategy;
use super::target::QuantTarget;

/// Builder for constructing a [`QuantStrategy`] incrementally.
pub struct QuantStrategyBuilder {
    inner: QuantStrategy,
}

impl QuantStrategyBuilder {
    /// Start from the default strategy (Q8_0 for all layers).
    pub fn new() -> Self {
        Self { inner: QuantStrategy::default() }
    }

    /// Set the attention quantization target.
    pub fn attention(mut self, target: QuantTarget) -> Self {
        self.inner.attention = target;
        self
    }

    /// Set the feed-forward quantization target.
    pub fn feed_forward(mut self, target: QuantTarget) -> Self {
        self.inner.feed_forward = target;
        self
    }

    /// Set the output quantization target.
    pub fn output(mut self, target: QuantTarget) -> Self {
        self.inner.output = target;
        self
    }

    /// Set the MoE quantization target.
    pub fn moe(mut self, target: QuantTarget) -> Self {
        self.inner.moe = target;
        self
    }

    /// Set the gate quantization target.
    pub fn gate(mut self, target: QuantTarget) -> Self {
        self.inner.gate = target;
        self
    }

    /// Set the minimum dimension for quantization.
    pub fn min_dim(mut self, dim: usize) -> Self {
        self.inner.min_dim = dim;
        self
    }

    /// Build the final [`QuantStrategy`].
    pub fn build(self) -> QuantStrategy {
        self.inner
    }
}

impl Default for QuantStrategyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: QuantStrategyBuilder::build
    #[test]
    fn test_builder_default_matches_default_strategy() {
        let built = QuantStrategyBuilder::new().build();
        let default = QuantStrategy::default();
        assert_eq!(built.attention, default.attention);
        assert_eq!(built.feed_forward, default.feed_forward);
        assert_eq!(built.output, default.output);
        assert_eq!(built.moe, default.moe);
        assert_eq!(built.gate, default.gate);
        assert_eq!(built.min_dim, default.min_dim);
    }

    /// @covers: QuantStrategyBuilder::new
    #[test]
    fn test_new_starts_from_default() {
        let b = QuantStrategyBuilder::new();
        let s = b.build();
        assert_eq!(s.attention, QuantTarget::Q8_0);
    }

    /// @covers: QuantStrategyBuilder::attention
    #[test]
    fn test_attention_overrides_attention_target() {
        let s = QuantStrategyBuilder::new().attention(QuantTarget::None).build();
        assert_eq!(s.attention, QuantTarget::None);
        assert_eq!(s.feed_forward, QuantTarget::Q8_0);
    }

    /// @covers: QuantStrategyBuilder::feed_forward
    #[test]
    fn test_feed_forward_overrides_ff_target() {
        let s = QuantStrategyBuilder::new().feed_forward(QuantTarget::F16).build();
        assert_eq!(s.feed_forward, QuantTarget::F16);
    }

    /// @covers: QuantStrategyBuilder::output
    #[test]
    fn test_output_overrides_output_target() {
        let s = QuantStrategyBuilder::new().output(QuantTarget::None).build();
        assert_eq!(s.output, QuantTarget::None);
    }

    /// @covers: QuantStrategyBuilder::moe
    #[test]
    fn test_moe_overrides_moe_target() {
        let s = QuantStrategyBuilder::new().moe(QuantTarget::Q4_0).build();
        assert_eq!(s.moe, QuantTarget::Q4_0);
    }

    /// @covers: QuantStrategyBuilder::gate
    #[test]
    fn test_gate_overrides_gate_target() {
        let s = QuantStrategyBuilder::new().gate(QuantTarget::Q4_1).build();
        assert_eq!(s.gate, QuantTarget::Q4_1);
    }

    /// @covers: QuantStrategyBuilder::min_dim
    #[test]
    fn test_min_dim_sets_threshold() {
        let s = QuantStrategyBuilder::new().min_dim(2048).build();
        assert_eq!(s.min_dim, 2048);
    }

    /// @covers: QuantStrategyBuilder::build
    #[test]
    fn test_build_produces_strategy() {
        let s = QuantStrategyBuilder::new()
            .attention(QuantTarget::None)
            .output(QuantTarget::F16)
            .build();
        assert_eq!(s.attention, QuantTarget::None);
        assert_eq!(s.output, QuantTarget::F16);
    }
}
