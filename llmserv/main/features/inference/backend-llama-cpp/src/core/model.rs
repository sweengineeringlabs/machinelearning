//! `LlamaCppModel` + `LlamaCppTextCompleter`.
//!
//! The C/C++ boundary is crossed only through the `llama-cpp-2` safe
//! wrappers — no `unsafe` in this file.
//!
//! Lifecycle:
//!
//! * `LlamaBackend::init()` is called exactly once per process, owned by
//!   a `OnceLock`. Subsequent loads reuse the same backend handle.
//! * `LlamaModel` (weights) is loaded once at daemon startup and held
//!   for the daemon's lifetime. Cheap to borrow via `&self`.
//! * `LlamaContext` (KV cache, runtime state) is created per
//!   `TextCompleter` invocation — i.e. per incoming HTTP request —
//!   inside `run_decode` and dropped when the call returns. This is
//!   the fresh-per-call strategy from P7.B audit finding #3. A pooled
//!   strategy is a future optimization; correctness first.
//!
//! Sampling is greedy (argmax over logits). This honors `temperature
//! = 0.0` from `CompletionParams` exactly; temperature > 0 / top-k /
//! top-p are TODOs that will plug into the existing sampler in
//! `rustml-generation` so both backends share the same sampling
//! semantics.

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context as _, Result, anyhow, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llmbackend::{Model, ModelSource, ModelSpec};
use rustml_generation::{CompletionParams, GenerationError, GenerationResult, TextCompleter};
use rustml_inference_layers::PoolingStrategy;
use rustml_model::{ModelError, ModelResult, OptProfile};
use rustml_tokenizer::{Tokenizer, TokenizerError, TokenizerResult};

/// Process-wide `llama.cpp` backend handle. `llama_backend_init` may only
/// be called once per process — `OnceLock` enforces that.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn get_backend() -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let new_backend = LlamaBackend::init().context("llama_backend_init failed")?;
    match BACKEND.set(new_backend) {
        Ok(()) => Ok(BACKEND.get().expect("just set")),
        // Race: another thread beat us. Drop our init, use theirs.
        Err(_) => Ok(BACKEND.get().expect("race: another thread won init")),
    }
}

/// A loaded llama.cpp model. Holds weights for the daemon's lifetime;
/// contexts are created per-request inside the completer.
pub struct LlamaCppModel {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    template: Option<LlamaChatTemplate>,
    gguf_path: PathBuf,
    model_id: String,
    #[allow(dead_code)]
    profile: OptProfile,
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
        Box::new(LlamaCppTextCompleter {
            backend: self.backend,
            model: &self.model,
            template: self.template.as_ref(),
        })
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        // `LlamaCppModel` implements `Tokenizer` directly (below) — no
        // intermediate adapter allocation, no self-referential struct.
        self
    }

    fn embed(&self, _token_ids: &[u32], _strategy: PoolingStrategy) -> ModelResult<Vec<f32>> {
        Err(ModelError::Model(
            "embeddings are not supported by the llama_cpp backend; \
             configure backend = \"native_rust\" to serve /v1/embeddings"
                .into(),
        ))
    }
}

impl Tokenizer for LlamaCppModel {
    fn encode(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| TokenizerError::TokenizerError(format!("llama.cpp tokenize: {}", e)))?;
        Ok(tokens.into_iter().map(|t| t.0 as u32).collect())
    }

    fn decode(&self, tokens: &[u32]) -> TokenizerResult<String> {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        for &t in tokens {
            let piece = self
                .model
                .token_to_piece(LlamaToken(t as i32), &mut decoder, false, None)
                .map_err(|e| {
                    TokenizerError::TokenizerError(format!(
                        "llama.cpp token_to_piece({}): {}",
                        t, e
                    ))
                })?;
            out.push_str(&piece);
        }
        Ok(out)
    }

    fn vocab_size(&self) -> usize {
        self.model.n_vocab() as usize
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        // llama-cpp-2 doesn't expose a single-token reverse lookup.
        // Callers that need it should use `encode(&text)` and inspect
        // the resulting ids.
        None
    }
}

/// Per-request text completer. Builds a fresh `LlamaContext` inside
/// each decode call; dropped when the completer is dropped.
pub struct LlamaCppTextCompleter<'a> {
    backend: &'static LlamaBackend,
    model: &'a LlamaModel,
    template: Option<&'a LlamaChatTemplate>,
}

impl LlamaCppTextCompleter<'_> {
    /// Core decode loop. Tokenizes the prompt, runs `llama.cpp`'s
    /// forward pass, samples greedily, invokes the callback per token,
    /// returns the full generated text (without the prompt).
    fn run_decode(
        &self,
        prompt: &str,
        params: &CompletionParams,
        mut callback: Option<&mut dyn FnMut(u32) -> bool>,
    ) -> GenerationResult<String> {
        let prompt_tokens = self.model.str_to_token(prompt, AddBos::Always).map_err(|e| {
            GenerationError::Generation(format!("llama.cpp tokenize failed: {}", e))
        })?;

        if prompt_tokens.is_empty() {
            return Err(GenerationError::Generation(
                "llama.cpp: empty prompt after tokenization".into(),
            ));
        }

        // Size the context window to hold the prompt plus generation
        // headroom. Minimum 2048 to keep small prompts fast.
        let n_ctx_request = ((prompt_tokens.len() + params.max_tokens + 64) as u32).max(2048);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx_request));
        let mut ctx = self
            .model
            .new_context(self.backend, ctx_params)
            .map_err(|e| {
                GenerationError::Generation(format!("llama.cpp new_context failed: {}", e))
            })?;

        if prompt_tokens.len() + params.max_tokens >= ctx.n_ctx() as usize {
            return Err(GenerationError::Generation(format!(
                "llama.cpp: prompt ({}) + max_tokens ({}) exceeds context window ({})",
                prompt_tokens.len(),
                params.max_tokens,
                ctx.n_ctx()
            )));
        }

        let mut batch = LlamaBatch::new(ctx.n_batch() as usize, 1);
        let last_prompt_idx = prompt_tokens.len() - 1;
        for (i, tok) in prompt_tokens.iter().enumerate() {
            let emit_logits = i == last_prompt_idx;
            batch
                .add(*tok, i as i32, &[0], emit_logits)
                .map_err(|e| {
                    GenerationError::Generation(format!("llama.cpp batch.add(prompt): {}", e))
                })?;
        }
        ctx.decode(&mut batch).map_err(|e| {
            GenerationError::Generation(format!("llama.cpp decode(prompt): {}", e))
        })?;

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        let mut pos = prompt_tokens.len() as i32;

        for _step in 0..params.max_tokens {
            // Greedy sample: argmax over the last-position logits.
            let logits = ctx.get_logits_ith(batch.n_tokens() - 1);
            let next_id = argmax(logits);
            let next_tok = LlamaToken(next_id);

            if self.model.is_eog_token(next_tok) {
                break;
            }

            let piece = self
                .model
                .token_to_piece(next_tok, &mut decoder, false, None)
                .map_err(|e| {
                    GenerationError::Generation(format!("llama.cpp token_to_piece: {}", e))
                })?;
            out.push_str(&piece);

            if let Some(cb) = callback.as_deref_mut() {
                if !cb(next_id as u32) {
                    break;
                }
            }

            if let Some(deadline) = params.deadline {
                if std::time::Instant::now() >= deadline {
                    break;
                }
            }

            batch.clear();
            batch.add(next_tok, pos, &[0], true).map_err(|e| {
                GenerationError::Generation(format!("llama.cpp batch.add(step): {}", e))
            })?;
            ctx.decode(&mut batch).map_err(|e| {
                GenerationError::Generation(format!("llama.cpp decode(step): {}", e))
            })?;
            pos += 1;
        }

        Ok(out)
    }
}

impl TextCompleter for LlamaCppTextCompleter<'_> {
    fn complete(
        &mut self,
        prompt: &str,
        params: &CompletionParams,
    ) -> GenerationResult<String> {
        self.run_decode(prompt, params, None)
    }

    fn complete_stream(
        &mut self,
        prompt: &str,
        params: &CompletionParams,
        callback: &mut dyn FnMut(u32) -> bool,
    ) -> GenerationResult<String> {
        self.run_decode(prompt, params, Some(callback))
    }

    fn complete_turn_stream(
        &mut self,
        messages: &[(&str, &str)],
        params: &CompletionParams,
        callback: &mut dyn FnMut(u32) -> bool,
    ) -> GenerationResult<String> {
        let template = self.template.ok_or_else(|| {
            GenerationError::Generation(
                "llama_cpp backend: GGUF file has no embedded chat template; \
                 /v1/chat/completions requires one. Use /v1/completions with a \
                 pre-formatted prompt instead."
                    .into(),
            )
        })?;

        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|(role, content)| {
                LlamaChatMessage::new((*role).to_string(), (*content).to_string())
            })
            .collect::<Result<_, _>>()
            .map_err(|e| {
                GenerationError::Generation(format!("llama.cpp chat message: {}", e))
            })?;

        let prompt = self
            .model
            .apply_chat_template(template, &chat, /* add_ass = */ true)
            .map_err(|e| {
                GenerationError::Generation(format!("llama.cpp apply_chat_template: {}", e))
            })?;

        self.run_decode(&prompt, params, Some(callback))
    }
}

/// Greedy sampler. Returns the token id with the highest logit.
fn argmax(logits: &[f32]) -> i32 {
    let mut best_i: i32 = 0;
    let mut best_v: f32 = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as i32;
        }
    }
    best_i
}

/// Construct an `LlamaCppModel` from a parsed `ModelSpec`.
pub fn load_llama_cpp_model(
    spec: &ModelSpec,
    profile: OptProfile,
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

    let backend = get_backend()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(backend, &gguf_path, &model_params)
        .with_context(|| format!("llama.cpp load_from_file: {}", gguf_path.display()))?;

    // Chat template is optional — some GGUF files ship without one.
    // Keep as Option; surface a clear error only if someone calls
    // `complete_turn_stream` on a template-less model.
    let template = model.chat_template(None).ok();

    let model_id = gguf_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "llama-cpp-model".into());

    log::info!(
        "llama_cpp backend: loaded '{}' (profile={:?}, path={}, template={})",
        model_id,
        profile,
        gguf_path.display(),
        if template.is_some() { "yes" } else { "absent" }
    );

    Ok(LlamaCppModel {
        backend,
        model,
        template,
        gguf_path,
        model_id,
        profile,
    })
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

    fn err_msg(r: Result<LlamaCppModel>) -> String {
        match r {
            Ok(_) => panic!("expected error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn test_load_rejects_safetensors_source_with_clear_error() {
        let spec = spec_with(ModelSource::Safetensors, None);
        let msg = err_msg(load_llama_cpp_model(&spec, OptProfile::Optimized, ""));
        assert!(msg.contains("source = \"gguf\""), "got: {}", msg);
    }

    #[test]
    fn test_load_rejects_missing_path_with_clear_error() {
        let spec = spec_with(ModelSource::Gguf, None);
        let msg = err_msg(load_llama_cpp_model(&spec, OptProfile::Optimized, ""));
        assert!(msg.contains("path"), "got: {}", msg);
    }

    #[test]
    fn test_load_rejects_nonexistent_file_with_clear_error() {
        let spec = spec_with(ModelSource::Gguf, Some("/definitely/does/not/exist.gguf"));
        let msg = err_msg(load_llama_cpp_model(&spec, OptProfile::Optimized, ""));
        assert!(msg.contains("not found"), "got: {}", msg);
    }

    #[test]
    fn test_argmax_picks_highest_logit() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3, 0.2]), 1);
        assert_eq!(argmax(&[3.0, -1.0, 2.0]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY, 0.0]), 1);
    }
}
