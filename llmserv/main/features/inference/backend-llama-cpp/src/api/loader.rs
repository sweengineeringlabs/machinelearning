//! Public surface: the `ModelBackendLoader` implementation the daemon
//! registers under `ModelBackend::LlamaCpp`.

use anyhow::Result;
use llmbackend::{Model, ModelBackendLoader, ModelSpec};
use swe_llmmodel_model::OptProfile;

use crate::core::model::load_llama_cpp_model;

/// Loads GGUF models via `llama-cpp-2`.
///
/// Requires `[model].source = "gguf"` and `[model].path` in
/// application.toml. SafeTensors loading is not supported here — callers
/// should use the native-Rust backend for that.
pub struct LlamaCppBackendLoader;

impl ModelBackendLoader for LlamaCppBackendLoader {
    fn name(&self) -> &'static str {
        "llama_cpp"
    }

    fn load(
        &self,
        spec: &ModelSpec,
        profile: OptProfile,
        merged_toml: &str,
    ) -> Result<Box<dyn Model>> {
        load_llama_cpp_model(spec, profile, merged_toml)
    }
}
