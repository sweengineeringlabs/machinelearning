//! SAF (single application facade) re-exports for `quant-engine`.
//!
//! Downstream crates depend on `quant_engine::DefaultQuantService` — that
//! name is routed here from `core::quantizer::default`. Everything else is
//! crate-internal.

pub use crate::core::quantizer::default::DefaultQuantService;
