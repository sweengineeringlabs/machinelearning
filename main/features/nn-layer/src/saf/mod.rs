// Public facade — re-exports the full API surface.

// Error types
pub use crate::api::error::{NnLayerError, NnLayerResult};

// Traits
pub use crate::api::traits::{Activation, Norm};

// Configuration types
pub use crate::api::types::NormConfig;

// Default implementations
pub use crate::core::activations::{Gelu, Silu};
pub use crate::core::layer_norm::DefaultLayerNorm;
pub use crate::core::rms_norm::DefaultRmsNorm;
