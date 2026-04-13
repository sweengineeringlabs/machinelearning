//! Model-selection config schema. Parsed from `[model]` section of
//! `application.toml`. Lives here (not in the daemon) so backend crates
//! can accept a `&ModelSpec` without pulling the daemon in.

use serde::Deserialize;

/// Which model to load and from where.
#[derive(Debug, Deserialize)]
pub struct ModelSpec {
    /// Inference backend that executes the forward pass.
    #[serde(default)]
    pub backend: ModelBackend,
    #[serde(default)]
    pub source: ModelSource,
    /// HuggingFace repo ID (required when source = "safetensors").
    #[serde(default)]
    pub id: Option<String>,
    /// Local GGUF path (required when source = "gguf").
    #[serde(default)]
    pub path: Option<String>,
}

impl Default for ModelSpec {
    fn default() -> Self {
        Self {
            backend: ModelBackend::default(),
            source: ModelSource::default(),
            id: None,
            path: None,
        }
    }
}

/// Where the model weights come from.
#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    #[default]
    Safetensors,
    Gguf,
}

/// Inference backend selection. Pairs with `ModelSource` to pick a loader
/// at daemon startup. Each variant has a registered `ModelBackendLoader`.
#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ModelBackend {
    /// Native Rust forward pass (rustml-model + rustml-inference-layers).
    #[default]
    NativeRust,
    /// llama.cpp-backed forward pass. Requires the daemon to be built with
    /// the `backend-llama-cpp` cargo feature.
    LlamaCpp,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_spec_defaults_to_native_rust_backend_when_backend_key_absent() {
        let toml = r#"
            source = "safetensors"
            id = "x"
        "#;
        let spec: ModelSpec = toml::from_str(toml).unwrap();
        assert_eq!(spec.backend, ModelBackend::NativeRust);
    }

    #[test]
    fn test_model_spec_parses_native_rust_backend_explicit() {
        let toml = r#"
            backend = "native_rust"
            source = "safetensors"
            id = "x"
        "#;
        let spec: ModelSpec = toml::from_str(toml).unwrap();
        assert_eq!(spec.backend, ModelBackend::NativeRust);
    }

    #[test]
    fn test_model_spec_parses_llama_cpp_backend() {
        let toml = r#"
            backend = "llama_cpp"
            source = "gguf"
            path = "x.gguf"
        "#;
        let spec: ModelSpec = toml::from_str(toml).unwrap();
        assert_eq!(spec.backend, ModelBackend::LlamaCpp);
        assert_eq!(spec.source, ModelSource::Gguf);
        assert_eq!(spec.path.as_deref(), Some("x.gguf"));
    }

    #[test]
    fn test_model_spec_unknown_backend_rejected() {
        let toml = r#"
            backend = "tensorflow"
            source = "gguf"
            path = "x.gguf"
        "#;
        let err = toml::from_str::<ModelSpec>(toml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("tensorflow") || msg.contains("variant"),
            "unexpected error: {}", msg
        );
    }
}
