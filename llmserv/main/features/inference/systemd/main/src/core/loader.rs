use std::path::PathBuf;

use anyhow::{Result, anyhow};

use swe_llmmodel_loader::{DefaultLoader, LoadModel};
use swe_llmmodel_model::OptProfile;

use llmbackend::{Model, ModelBackendLoader, ModelSource, ModelSpec};

use super::state::DefaultModel;

/// Native Rust backend loader. Dispatches on `[model].source` to the
/// SafeTensors or GGUF path of [`swe_llmmodel_loader::DefaultLoader`],
/// then wraps the resulting `LoadedModel` into the daemon's
/// [`DefaultModel`] via `From`.
pub struct NativeRustBackendLoader;

impl ModelBackendLoader for NativeRustBackendLoader {
    fn name(&self) -> &'static str {
        "native_rust"
    }

    fn load(
        &self,
        spec: &ModelSpec,
        profile: OptProfile,
        merged_toml: &str,
    ) -> Result<Box<dyn Model>> {
        let loader = DefaultLoader::new();
        let loaded = match spec.source {
            ModelSource::Safetensors => {
                let id = spec.id.as_deref().ok_or_else(|| {
                    anyhow!("[model].source = \"safetensors\" requires [model].id")
                })?;
                loader.load_safetensors(id, profile, merged_toml)?
            }
            ModelSource::Gguf => {
                let path_str = spec.path.as_deref().ok_or_else(|| {
                    anyhow!("[model].source = \"gguf\" requires [model].path")
                })?;
                loader.load_gguf(&PathBuf::from(path_str), profile)?
            }
        };
        let default: DefaultModel = loaded.into();
        Ok(Box::new(default))
    }
}
