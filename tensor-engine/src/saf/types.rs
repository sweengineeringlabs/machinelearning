//! Re-exports of core types for the public surface.
//!
//! This module lives in saf/ and is allowed to access core/ directly.
//! It forms the bridge between internal types and the public API.
//!
//! Types that are pub(crate) in core/ are NOT re-exported here.
//! They are accessed only through wrapper functions in saf/wrappers.rs.

pub use crate::core::tensor::{Tensor, TensorBuilder, Storage, f32_vec_to_bytes, f32_slice_to_bytes};
pub use crate::core::shape_mod::shape::Shape;
pub use crate::core::runtime::opt_profile::OptProfile;
pub use crate::api::quant_target_api::QuantTarget;

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
}
