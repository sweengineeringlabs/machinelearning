use std::path::Path;

use swe_llmmodel_model::OptProfile;

use crate::api::error::LoaderResult;
use crate::api::types::LoadedModel;

/// End-to-end model load contract.
///
/// Implementations compose download + format I/O + weight mapping +
/// architecture build + tokenizer + runtime quantization into a
/// single [`LoadedModel`].
pub trait LoadModel {
    /// Resolve a HuggingFace model id to a runnable `LoadedModel`.
    /// `quantization_toml` holds the merged `[quantization]` config that
    /// drives the runtime quantizer; empty string means no quantization.
    fn load_safetensors(
        &self,
        model_id: &str,
        profile: OptProfile,
        quantization_toml: &str,
    ) -> LoaderResult<LoadedModel>;

    /// Load a model from a local GGUF file.
    fn load_gguf(&self, path: &Path, profile: OptProfile) -> LoaderResult<LoadedModel>;
}
