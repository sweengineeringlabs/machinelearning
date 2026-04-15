//! SAF (single application facade) re-exports for `quant-io`.
//!
//! The only public symbol is `ModelIO`, the namespace type for
//! safetensors + GGUF read/write. Downstream crates depend on
//! `quant_io::ModelIO`.

pub use crate::core::io::model::ModelIO;
