//! # swe-llmmodel-weights
//!
//! Maps HuggingFace tensor names to architecture-internal names, and
//! validates that the expected set of weights is present.

pub mod api;
mod core;
mod saf;

pub use saf::*;
