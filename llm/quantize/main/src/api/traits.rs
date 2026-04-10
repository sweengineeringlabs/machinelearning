use crate::api::error::QuantizeResult;
use crate::api::types::{QuantizeConfig, QuantizeReport};

/// Quantization engine trait.
pub trait QuantizeEngine: Send + Sync {
    /// Run the full quantization pipeline: load, quantize, export to GGUF.
    fn run(&self, config: &QuantizeConfig) -> QuantizeResult<QuantizeReport>;
}
