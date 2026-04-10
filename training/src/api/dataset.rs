use crate::api::error::SwetsResult;
use crate::api::tensor::Tensor;

/// A dataset that produces indexed (input, target) tensor pairs.
///
/// Implement this trait for any data source to use it with `DataLoader`.
pub trait Dataset {
    /// Number of samples in the dataset.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the (input, target) pair at the given index.
    fn get(&self, index: usize) -> SwetsResult<(Tensor, Tensor)>;

    /// Shape of a single input sample (e.g., [window_size, num_features]).
    fn input_shape(&self) -> Vec<usize>;

    /// Dimensionality of the target (e.g., 1 for single-target regression).
    fn target_dim(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: Dataset::is_empty
    #[test]
    fn test_is_empty_default_delegates_to_len() {
        struct EmptyDs;
        impl Dataset for EmptyDs {
            fn len(&self) -> usize { 0 }
            fn get(&self, _: usize) -> SwetsResult<(Tensor, Tensor)> { unimplemented!() }
            fn input_shape(&self) -> Vec<usize> { vec![1] }
            fn target_dim(&self) -> usize { 1 }
        }
        assert!(EmptyDs.is_empty());
    }
}
