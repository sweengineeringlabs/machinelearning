//! # swe-llmmodel-download
//!
//! Model download and cache management. Wraps HuggingFace Hub with both
//! async (reqwest) and sync (hf-hub) paths. Returns file paths only —
//! format parsing lives in `swe-llmmodel-io`.

pub mod api;
mod core;
mod saf;

pub use saf::*;
