pub use crate::api::error::*;
pub use crate::api::traits::*;
pub use crate::api::types::*;

use crate::core::engine::DefaultQuantizeEngine;

/// Create a new quantization engine.
pub fn create_engine() -> impl QuantizeEngine {
    DefaultQuantizeEngine::new()
}
