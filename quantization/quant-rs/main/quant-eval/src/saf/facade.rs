//! SAF (single application facade) re-exports for `quant-eval`.
//!
//! The public surface is intentionally narrow: callers depend on the
//! evaluator and its metric bundle, nothing else.

pub use crate::core::eval::service::{EvalService, Metrics};
