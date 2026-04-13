//! `LlamaCppModel` — the `llmbackend::Model` implementation backed by
//! `llama-cpp-2`.
//!
//! ## Status: architectural skeleton
//!
//! This file defines the type surface and wires it to the daemon's
//! `Model` trait. `load_llama_cpp_model` currently returns a structured
//! error explaining the llama-cpp-2 integration is pending. The trait
//! impl is compile-verified but unreachable at runtime until the
//! follow-up lands.
//!
//! The split is deliberate:
//! - architectural wiring and feature flags can be reviewed and tested
//!   today without a C++ toolchain
//! - llama-cpp-2 calls (model load, context, decode, logits,
//!   tokenization, chat templating) land in a separate commit once
//!   MSVC is available on the dev box for local verification

use anyhow::{Result, anyhow, bail};
use llmbackend::{Model, ModelSource, ModelSpec};
use rustml_generation::{CompletionParams, GenerationError, GenerationResult, TextCompleter};
use rustml_inference_layers::PoolingStrategy;
use rustml_model::{ModelError, ModelResult, OptProfile};
use rustml_tokenizer::{Tokenizer, TokenizerError, TokenizerResult};
use std::path::PathBuf;

/// Handle to a loaded llama.cpp model. Currently inert — construction
/// is blocked at `load_llama_cpp_model` until the real binding lands.
#[derive(Debug)]
pub struct LlamaCppModel {
    model_id: String,
    gguf_path: PathBuf,
    #[allow(dead_code)]
    profile: OptProfile,
    #[allow(dead_code)]
    tokenizer: LlamaCppTokenizerAdapter,
}

impl LlamaCppModel {
    pub fn gguf_path(&self) -> &PathBuf {
        &self.gguf_path
    }
}

impl Model for LlamaCppModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn open_text_completer(&self) -> Box<dyn TextCompleter + '_> {
        Box::new(LlamaCppTextCompleter::new(&self.model_id))
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        &self.tokenizer
    }

    fn embed(&self, _token_ids: &[u32], _strategy: PoolingStrategy) -> ModelResult<Vec<f32>> {
        Err(ModelError::Model(
            "embeddings are not supported by the llama_cpp backend; \
             configure backend = \"native_rust\" to serve /v1/embeddings"
                .into(),
        ))
    }
}

/// Tokenizer wrapper that will delegate to llama.cpp's BPE once the
/// binding lands. Methods return clear errors until then.
#[derive(Debug)]
pub struct LlamaCppTokenizerAdapter {
    model_id: String,
}

impl Tokenizer for LlamaCppTokenizerAdapter {
    fn encode(&self, _text: &str) -> TokenizerResult<Vec<u32>> {
        Err(TokenizerError::TokenizerError(format!(
            "LlamaCppTokenizerAdapter::encode for '{}' is not yet implemented \
             — llama-cpp-2 integration is a follow-up subtask",
            self.model_id
        )))
    }

    fn decode(&self, _tokens: &[u32]) -> TokenizerResult<String> {
        Err(TokenizerError::TokenizerError(format!(
            "LlamaCppTokenizerAdapter::decode for '{}' is not yet implemented \
             — llama-cpp-2 integration is a follow-up subtask",
            self.model_id
        )))
    }

    fn vocab_size(&self) -> usize {
        0
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
}

/// Per-request text completer. Holds no llama.cpp state today (the real
/// impl will own a per-call `LlamaContext` + sampling chain).
pub struct LlamaCppTextCompleter {
    model_id: String,
}

impl LlamaCppTextCompleter {
    pub fn new(model_id: &str) -> Self {
        Self { model_id: model_id.to_string() }
    }
}

impl TextCompleter for LlamaCppTextCompleter {
    fn complete(
        &mut self,
        _prompt: &str,
        _params: &CompletionParams,
    ) -> GenerationResult<String> {
        Err(GenerationError::Generation(format!(
            "llama_cpp backend completion for '{}' is not yet implemented \
             — llama-cpp-2 integration is a follow-up subtask; \
             configure backend = \"native_rust\" for working completions",
            self.model_id
        )))
    }

    fn complete_stream(
        &mut self,
        _prompt: &str,
        _params: &CompletionParams,
        _callback: &mut dyn FnMut(u32) -> bool,
    ) -> GenerationResult<String> {
        Err(GenerationError::Generation(format!(
            "llama_cpp backend streaming completion for '{}' is not yet implemented",
            self.model_id
        )))
    }

    fn complete_turn_stream(
        &mut self,
        _messages: &[(&str, &str)],
        _params: &CompletionParams,
        _callback: &mut dyn FnMut(u32) -> bool,
    ) -> GenerationResult<String> {
        Err(GenerationError::Generation(format!(
            "llama_cpp backend chat completion for '{}' is not yet implemented",
            self.model_id
        )))
    }
}

/// Construct an `LlamaCppModel` from a parsed `ModelSpec`.
///
/// Validates inputs eagerly (wrong source, missing path, missing file)
/// and then refuses to proceed further until the llama-cpp-2 integration
/// lands. This means daemon startup fails fast with a clear message if
/// someone configures `backend = "llama_cpp"` before the follow-up ships
/// — rather than booting and returning errors on every request.
pub fn load_llama_cpp_model(
    spec: &ModelSpec,
    _profile: OptProfile,
    _merged_toml: &str,
) -> Result<LlamaCppModel> {
    if !matches!(spec.source, ModelSource::Gguf) {
        bail!(
            "backend = \"llama_cpp\" requires source = \"gguf\" (got {:?}). \
             SafeTensors loading is not supported by this backend — switch \
             to backend = \"native_rust\" to load SafeTensors models.",
            spec.source
        );
    }
    let path_str = spec
        .path
        .as_deref()
        .ok_or_else(|| anyhow!("backend = \"llama_cpp\" requires [model].path"))?;
    let gguf_path = PathBuf::from(path_str);
    if !gguf_path.exists() {
        bail!(
            "GGUF file not found for llama_cpp backend: {}",
            gguf_path.display()
        );
    }

    bail!(
        "backend = \"llama_cpp\" is registered but its llama-cpp-2 integration \
         is not yet implemented. The GGUF file at '{}' was validated. The \
         daemon is refusing to start rather than booting a broken backend. \
         Track the follow-up subtask in llmserv/BACKLOG.md (P7.B step 4 \
         completion). For working inference today, set \
         [model].backend = \"native_rust\".",
        gguf_path.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_with(source: ModelSource, path: Option<&str>) -> ModelSpec {
        ModelSpec {
            backend: llmbackend::ModelBackend::LlamaCpp,
            source,
            id: None,
            path: path.map(String::from),
        }
    }

    #[test]
    fn test_load_rejects_safetensors_source_with_clear_error() {
        let spec = spec_with(ModelSource::Safetensors, None);
        let err = load_llama_cpp_model(&spec, OptProfile::Optimized, "")
            .expect_err("should reject safetensors source");
        let msg = err.to_string();
        assert!(
            msg.contains("source = \"gguf\""),
            "error should explain gguf requirement, got: {}", msg
        );
    }

    #[test]
    fn test_load_rejects_missing_path_with_clear_error() {
        let spec = spec_with(ModelSource::Gguf, None);
        let err = load_llama_cpp_model(&spec, OptProfile::Optimized, "")
            .expect_err("should reject missing path");
        let msg = err.to_string();
        assert!(msg.contains("path"), "error should mention path, got: {}", msg);
    }

    #[test]
    fn test_load_rejects_nonexistent_file_with_clear_error() {
        let spec = spec_with(ModelSource::Gguf, Some("/definitely/does/not/exist.gguf"));
        let err = load_llama_cpp_model(&spec, OptProfile::Optimized, "")
            .expect_err("should reject missing file");
        let msg = err.to_string();
        assert!(
            msg.contains("not found"),
            "error should explain file missing, got: {}", msg
        );
    }

    #[test]
    fn test_load_returns_pending_integration_error_for_valid_input() {
        // Use an existing file to pass the pre-checks.
        let tmp = std::env::temp_dir().join("llama_cpp_backend_test.gguf");
        std::fs::write(&tmp, b"fake-gguf-bytes-for-skeleton-test").unwrap();
        let spec = spec_with(ModelSource::Gguf, Some(tmp.to_str().unwrap()));

        let err = load_llama_cpp_model(&spec, OptProfile::Optimized, "")
            .expect_err("skeleton must refuse to boot until integration lands");
        let msg = err.to_string();
        assert!(
            msg.contains("not yet implemented") || msg.contains("llama-cpp-2"),
            "error should clearly state pending integration, got: {}", msg
        );

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_tokenizer_adapter_returns_clear_error() {
        let tok = LlamaCppTokenizerAdapter { model_id: "test-model".into() };
        let err = tok.encode("hello").expect_err("encode should error");
        assert!(err.to_string().contains("not yet implemented"));

        let err = tok.decode(&[1, 2, 3]).expect_err("decode should error");
        assert!(err.to_string().contains("not yet implemented"));

        assert_eq!(tok.vocab_size(), 0);
        assert_eq!(tok.token_to_id("x"), None);
    }

    #[test]
    fn test_text_completer_returns_clear_error_on_all_methods() {
        let mut c = LlamaCppTextCompleter::new("test-model");
        let params = CompletionParams::new(0.0, 16);

        let err = c.complete("hello", &params).expect_err("complete should error");
        assert!(err.to_string().contains("not yet implemented"));

        let mut cb = |_tok: u32| true;
        let err = c.complete_stream("hello", &params, &mut cb)
            .expect_err("complete_stream should error");
        assert!(err.to_string().contains("not yet implemented"));

        let err = c.complete_turn_stream(&[("user", "hi")], &params, &mut cb)
            .expect_err("complete_turn_stream should error");
        assert!(err.to_string().contains("not yet implemented"));
    }
}
