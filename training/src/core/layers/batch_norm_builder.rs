use super::batch_norm::BatchNorm1d;

/// Builder for `BatchNorm1d` with optional epsilon and momentum.
pub struct BatchNorm1dBuilder {
    num_features: usize,
    eps: f32,
    momentum: f32,
}

impl BatchNorm1dBuilder {
    /// Creates a new builder with the required `num_features`.
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
        }
    }

    /// Sets the epsilon value for numerical stability.
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the momentum for running statistics update.
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Builds the `BatchNorm1d` layer with the configured parameters.
    pub fn build(self) -> BatchNorm1d {
        BatchNorm1d::with_config(self.num_features, self.eps, self.momentum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: BatchNorm1dBuilder::new
    #[test]
    fn test_builder_new_uses_defaults() {
        let builder = BatchNorm1dBuilder::new(4);
        let bn = builder.build();
        assert_eq!(bn.num_features(), 4);
        assert!((bn.eps() - 1e-5).abs() < 1e-10);
        assert!((bn.momentum() - 0.1).abs() < 1e-6);
    }

    /// @covers: BatchNorm1dBuilder::build
    #[test]
    fn test_builder_custom_eps_and_momentum() {
        let bn = BatchNorm1dBuilder::new(8)
            .eps(1e-3)
            .momentum(0.2)
            .build();
        assert_eq!(bn.num_features(), 8);
        assert!((bn.eps() - 1e-3).abs() < 1e-10);
        assert!((bn.momentum() - 0.2).abs() < 1e-6);
    }

    /// @covers: BatchNorm1dBuilder::eps
    #[test]
    fn test_builder_eps_setter() {
        let bn = BatchNorm1dBuilder::new(2).eps(1e-4).build();
        assert!((bn.eps() - 1e-4).abs() < 1e-10);
    }

    /// @covers: BatchNorm1dBuilder::momentum
    #[test]
    fn test_builder_momentum_setter() {
        let bn = BatchNorm1dBuilder::new(2).momentum(0.05).build();
        assert!((bn.momentum() - 0.05).abs() < 1e-6);
    }
}
