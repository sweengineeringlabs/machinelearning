//! Re-exports of core types for the public surface.
//!
//! This module lives in saf/ and is allowed to access core/ directly.
//! It forms the bridge between internal types and the public API.

pub use crate::core::tensor::{Tensor, TensorBuilder, Storage, f32_vec_to_bytes, f32_slice_to_bytes};
pub use crate::core::shape_mod::shape::Shape;
pub use crate::core::arena::tensor_pool::TensorPool;
pub use crate::core::runtime::runtime_config::RuntimeConfig;
pub use crate::core::runtime::opt_profile::OptProfile;
pub use crate::core::runtime::quant::strategy::QuantStrategy;
pub use crate::core::runtime::quant::strategy_builder::QuantStrategyBuilder;
pub use crate::core::runtime::quant::target::QuantTarget;

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: types re-exports
    #[test]
    fn test_reexported_tensor_is_constructible() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
    }

    /// @covers: types re-exports
    #[test]
    fn test_reexported_shape_is_constructible() {
        let s = Shape::new(vec![4, 5]);
        assert_eq!(s.ndim(), 2);
    }

    /// @covers: types re-exports
    #[test]
    fn test_reexported_tensor_pool_is_constructible() {
        let pool = TensorPool::new(8);
        assert!(pool.is_empty());
    }
}
