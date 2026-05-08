//! SPI for constructing a `Model` from parsed config.
//!
//! The daemon registers one `ModelBackendLoader` per `ModelBackend`
//! variant at startup. The loader owns all source-branching logic
//! (safetensors vs gguf vs future formats) internally — the daemon
//! dispatches once on the `ModelBackend` enum and delegates.
//!
//! Loaders return `Box<dyn Model>` and never expose backend-internal
//! types. See `crate::api::model::Model`.

use anyhow::Result;
use swe_llmmodel_model::OptProfile;

use crate::api::config::ModelSpec;
use crate::api::model::Model;

/// Loads a model using a specific backend (native Rust, llama.cpp, ...).
pub trait ModelBackendLoader: Send + Sync {
    /// Human-readable backend name for startup logs.
    fn name(&self) -> &'static str;

    /// Construct the model.
    ///
    /// `merged_toml` carries the full application.toml so loaders can
    /// read their own config sections (e.g. native-Rust reads
    /// `[quantization]`) without re-reading files.
    fn load(
        &self,
        spec: &ModelSpec,
        profile: OptProfile,
        merged_toml: &str,
    ) -> Result<Box<dyn Model>>;
}
