// SAF — public surface re-exports

// API types
pub use crate::api::dtype::*;
pub use crate::api::device::*;
pub use crate::api::error::*;
pub use crate::api::traits::*;

// Core types exposed via saf (implementation types needed by consumers)
pub use crate::core::tensor::{Tensor, Storage, f32_vec_to_bytes, f32_slice_to_bytes};
pub use crate::core::shape_mod::shape::Shape;
pub use crate::core::arena::tensor_pool::TensorPool;
pub use crate::core::runtime::runtime_config::RuntimeConfig;
pub use crate::core::runtime::opt_profile::OptProfile;
pub use crate::core::runtime::quant_strategy::QuantStrategy;
pub use crate::core::runtime::quant_target::QuantTarget;
