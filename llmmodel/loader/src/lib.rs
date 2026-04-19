//! # swe-llmmodel-loader
//!
//! End-to-end model loader. Composes every other llmmodel crate:
//! download (HF), io (safetensors), gguf (GGUF parser), tokenizer (BPE /
//! HF / GGUF), quantizer (runtime weight compression), model (registry +
//! config), and the architecture builders. The caller gets back a
//! [`LoadedModel`] — a ready-to-run [`LlmModel`](swe_llmmodel_model::LlmModel)
//! plus its tokenizer and decode-time metadata (chat template, EOS, BOS,
//! profile).

pub mod api;
mod core;
mod saf;

pub use saf::*;
