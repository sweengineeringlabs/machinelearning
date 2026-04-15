//! SAF (single application facade) re-exports for `quant-api`.
//!
//! Consumers depend on `quant_api::{Quantizer, QuantFormat, ...}` —
//! those names are routed here from `api::*`. Anything not re-exported
//! here is crate-internal.

pub use crate::api::error::{Error, QuantError};
pub use crate::api::format::QuantFormat;
pub use crate::api::quantizer::Quantizer;
pub use crate::api::tensor::QuantizedTensor;
