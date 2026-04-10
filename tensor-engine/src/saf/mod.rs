// SAF — public surface re-exports
//
// All re-exports come from api/ (never from core/ directly).
// Traits are NOT re-exported — standalone wrapper functions are provided instead.

mod types;
mod wrappers;

// API enum/error types
pub use crate::api::dtype::*;
pub use crate::api::device::*;
pub use crate::api::error::*;

// ConfigOps trait — allows .apply() on RuntimeConfig via trait method
pub use crate::api::config_ops::ConfigOps;

// Core types routed through saf/types
// TensorPool and QuantStrategy are pub(crate) — accessed via wrapper functions only
pub use types::{
    Tensor, TensorBuilder, Storage, f32_vec_to_bytes, f32_slice_to_bytes,
    Shape,
    OptProfile,
    QuantTarget,
};

// Wrapper functions (Rule 106 — standalone functions instead of trait re-exports)
pub use wrappers::{
    tensor_shape, tensor_dtype, tensor_matmul, tensor_add, tensor_softmax,
    apply_runtime_config, warmup_thread_pool, detect_simd,
    QuantConfig,
};
