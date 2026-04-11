/// Core traits used by this crate's model architectures.
///
/// These re-exports provide the canonical location for trait imports
/// within the architectures crate.

/// The `Layer` trait defines a neural network layer with forward pass
/// and parameter access.
pub use swe_ml_training::Layer;

/// The `Loss` trait defines a loss function for training.
pub use swe_ml_training::Loss;

/// The `Dataset` trait defines a data source for training/evaluation.
pub use swe_ml_training::Dataset;

/// Type alias for optimizer trait objects used in training loops.
pub type OptimizerBox = Box<dyn swe_ml_training::Optimizer>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_trait_is_reexported() {
        // Verify the Layer trait is accessible through this module
        fn accepts_layer<T: Layer>(_layer: &mut T) {}
        let mut linear = swe_ml_training::Linear::new(2, 1);
        accepts_layer(&mut linear);
    }
}
