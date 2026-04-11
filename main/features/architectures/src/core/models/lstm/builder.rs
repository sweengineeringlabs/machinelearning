use super::LSTM;

/// Fluent builder for constructing an [`LSTM`] instance.
///
/// Provides a step-by-step configuration interface with sensible defaults.
///
/// # Example
/// ```ignore
/// let lstm = LSTMBuilder::new(10, 20).num_layers(3).build();
/// ```
pub struct LSTMBuilder {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
}

impl LSTMBuilder {
    /// Start building an LSTM with required dimensions.
    ///
    /// Defaults to `num_layers = 1`.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            num_layers: 1,
        }
    }

    /// Set the number of stacked LSTM layers (default: 1).
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Consume the builder and construct the [`LSTM`].
    pub fn build(self) -> LSTM {
        LSTM::new(self.input_size, self.hidden_size, self.num_layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: build
    #[test]
    fn test_build_lstm_via_builder_default_layers() {
        let lstm = LSTMBuilder::new(10, 20).build();
        assert_eq!(lstm.input_size(), 10);
        assert_eq!(lstm.hidden_size(), 20);
        assert_eq!(lstm.num_layers(), 1);
    }

    /// @covers: num_layers
    #[test]
    fn test_build_lstm_via_builder_custom_layers() {
        let lstm = LSTMBuilder::new(5, 16).num_layers(3).build();
        assert_eq!(lstm.input_size(), 5);
        assert_eq!(lstm.hidden_size(), 16);
        assert_eq!(lstm.num_layers(), 3);
    }

    /// @covers: build
    #[test]
    fn test_new_creates_builder_with_defaults() {
        let builder = LSTMBuilder::new(8, 32);
        let lstm = builder.build();
        assert_eq!(lstm.num_layers(), 1);
    }

    /// @covers: num_layers
    #[test]
    fn test_num_layers_sets_layer_count() {
        let lstm = LSTMBuilder::new(4, 8).num_layers(5).build();
        assert_eq!(lstm.num_layers(), 5);
    }
}
