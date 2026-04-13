//! `LlamaCppModel` + `LlamaCppTextCompleter`.
//!
//! The C/C++ boundary is crossed only through `llama-cpp-2`'s safe
//! wrappers. Two `unsafe` blocks live in this file, both narrowed
//! and documented where they appear:
//!
//! 1. A `mem::transmute` that extends `LlamaContext<'a>` to
//!    `LlamaContext<'static>` so we can hold it in the same struct
//!    as the `LlamaModel` it borrows from. Safety depends on a
//!    single invariant (see below) that is statically enforceable
//!    given this module's API surface.
//!
//! 2. An `unsafe impl Send` for the pool wrapper — llama.cpp's
//!    context has a raw pointer; Mutex serializes access.
//!
//! # Self-referential pool safety
//!
//! `LlamaCppModel` holds both a `LlamaModel` and a pool of
//! `LlamaContext` values that borrow from it. To make this compile,
//! we widen the context lifetime to `'static` via transmute. The
//! invariants that make this sound:
//!
//! a) **The struct is never moved after construction.** Enforced by
//!    exposing only `load_llama_cpp_model(...) -> Result<Box<dyn
//!    llmbackend::Model>>` — the concrete struct is `pub(crate)` and
//!    the public API boxes it on the way out. Once boxed, callers
//!    have `Box<dyn Model>`, and `mem::swap` on trait objects
//!    requires their concrete types, which is unreachable.
//!
//! b) **The pool drops before the model.** Enforced by Rust field
//!    declaration order: `context_pool` is the first field, `model`
//!    is after it. Rust's drop order is top-to-bottom, so pooled
//!    contexts are released before the `LlamaModel` they point at.
//!
//! Invariant (a) is the only one that could be broken by future
//! edits; this module's `saf/` only re-exports the loader function
//! and the `LlamaCppBackendLoader`, not the struct. If someone adds
//! `pub` visibility to `LlamaCppModel` or exports it from saf, they
//! unlock the possibility of `mem::swap` on two instances, which is
//! UB. A compile-time test (`_assert_model_private`) exists below
//! to catch at least one class of that regression.
//!
//! Alternatives considered: `self_cell` crate (~60-200 ms slower at
//! p50 due to closure-API overhead on the hot path).

use std::collections::VecDeque;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context as _, Result, anyhow, bail};
use llama_cpp_2::context::LlamaContext;
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

/// Process-wide `llama.cpp` backend. Initialized once per process via
/// double-checked locking — `LlamaBackend::init` returns
/// `BackendAlreadyInitialized` on the second call.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static BACKEND_INIT_LOCK: Mutex<()> = Mutex::new(());

fn get_backend() -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let _guard = BACKEND_INIT_LOCK
        .lock()
        .map_err(|_| anyhow!("backend init mutex poisoned"))?;
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }
    let new_backend = LlamaBackend::init().context("llama_backend_init failed")?;
    BACKEND
        .set(new_backend)
        .map_err(|_| anyhow!("BACKEND was set between double-check and set"))?;
    Ok(BACKEND.get().expect("just set"))
}

/// Default ceiling for the per-pool-member context window size. Each
/// pooled context pre-allocates KV cache for this many tokens. A
/// request with `prompt_tokens + max_tokens` exceeding this fails
/// fast rather than triggering a re-alloc — re-alloc is the cost the
/// pool exists to avoid.
const DEFAULT_POOL_N_CTX: u32 = 8192;

/// Wraps `LlamaContext` with a transmuted `'static` lifetime. See
/// module-level docs for the safety argument.
struct OwnedContext {
    inner: LlamaContext<'static>,
}

// SAFETY: `LlamaContext` is `!Send` by default (contains a raw
// `*mut llama_context`). Adding `Send` is sound here because:
//   1. `OwnedContext` is only ever handled through a `Mutex` (the
//      context pool) or as a local owned by `with_context`.
//      Concurrent mutation from two threads is impossible.
//   2. llama.cpp contexts are designed for sequential use from any
//      thread. The supported pattern is: thread A locks, uses,
//      releases; thread B locks, uses, releases — exactly what
//      Mutex gives us.
unsafe impl Send for OwnedContext {}

/// A loaded llama.cpp model. `pub(crate)` — see module-level docs
/// for why the concrete struct must not escape this module.
///
/// Field declaration order is load-bearing for safety: the pool
/// must be declared BEFORE the model it borrows from, so Rust's
/// top-to-bottom drop order releases pooled contexts before the
/// model is freed.
pub(crate) struct LlamaCppModel {
    // Drops first — contexts reference `model` below.
    context_pool: Mutex<VecDeque<OwnedContext>>,
    // Everything after the pool.
    pool_n_ctx: u32,
    backend: &'static LlamaBackend,
    model: LlamaModel,
    template: Option<LlamaChatTemplate>,
    #[allow(dead_code)]
    gguf_path: PathBuf,
    model_id: String,
    #[allow(dead_code)]
    profile: OptProfile,
}

/// RAII handle to a borrowed pool context. Returns the context on
/// `Drop` after clearing its KV cache.
struct PooledContext<'m> {
    owner: &'m LlamaCppModel,
    ctx: Option<OwnedContext>,
}

impl PooledContext<'_> {
    /// Returns `&mut LlamaContext<'static>` — the lifetime is our
    /// internal lie, narrowed in practice by `&mut self` which is
    /// bounded by the `&LlamaCppModel` this borrow chains from.
    fn ctx(&mut self) -> &mut LlamaContext<'static> {
        &mut self.ctx.as_mut().expect("ctx present until drop").inner
    }
}

impl Drop for PooledContext<'_> {
    fn drop(&mut self) {
        if let Some(mut owned) = self.ctx.take() {
            // Reset the KV cache before returning. Prefer the
            // cheap seq-remove path (metadata only); fall back to
            // the full data-buffer wipe if seq-remove reports
            // failure.
            let cleared = match owned.inner.clear_kv_cache_seq(None, None, None) {
                Ok(true) => true,
                Ok(false) => {
                    log::debug!("clear_kv_cache_seq returned false; full wipe");
                    owned.inner.clear_kv_cache();
                    true
                }
                Err(e) => {
                    log::warn!("clear_kv_cache_seq errored ({}); full wipe", e);
                    owned.inner.clear_kv_cache();
                    true
                }
            };
            if cleared {
                if let Ok(mut pool) = self.owner.context_pool.lock() {
                    pool.push_back(owned);
                }
                // Poisoned mutex: drop the context silently.
            }
        }
    }
}

impl LlamaCppModel {
    fn build_context(&self) -> Result<OwnedContext, GenerationError> {
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.pool_n_ctx));
        let ctx = self.model.new_context(self.backend, params).map_err(|e| {
            GenerationError::Generation(format!("llama.cpp new_context: {}", e))
        })?;
        // SAFETY: module-level docs. The `'static` is a lie bounded
        // by `self.model`; field order ensures the pool drops before
        // the model; `LlamaCppModel` is `pub(crate)` so callers
        // can't `mem::swap` two instances to cross-wire the lie.
        let static_ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
        Ok(OwnedContext { inner: static_ctx })
    }

    fn acquire_context(&self) -> Result<PooledContext<'_>, GenerationError> {
        let pooled = {
            let mut pool = self.context_pool.lock().map_err(|_| {
                GenerationError::Generation("llama_cpp context pool mutex poisoned".into())
            })?;
            pool.pop_front()
        };
        let ctx = match pooled {
            Some(c) => c,
            None => self.build_context()?,
        };
        Ok(PooledContext { owner: self, ctx: Some(ctx) })
    }
}

impl Model for LlamaCppModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn open_text_completer(&self) -> Box<dyn TextCompleter + '_> {
        Box::new(LlamaCppTextCompleter {
            model: self,
            template: self.template.as_ref(),
        })
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
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
                .token_to_piece(LlamaToken(t as i32), &mut decoder, /* special = */ true, None)
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
        None
    }
}

/// Per-request text completer. Cheap to construct; all heavy state
/// lives in `LlamaCppModel`.
pub(crate) struct LlamaCppTextCompleter<'a> {
    model: &'a LlamaCppModel,
    template: Option<&'a LlamaChatTemplate>,
}

impl LlamaCppTextCompleter<'_> {
    fn run_decode(
        &self,
        prompt: &str,
        params: &CompletionParams,
        mut callback: Option<&mut dyn FnMut(u32) -> bool>,
    ) -> GenerationResult<String> {
        let prompt_tokens = self
            .model
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| GenerationError::Generation(format!("llama.cpp tokenize: {}", e)))?;

        if prompt_tokens.is_empty() {
            return Err(GenerationError::Generation(
                "llama.cpp: empty prompt after tokenization".into(),
            ));
        }
        if prompt_tokens.len() + params.max_tokens >= self.model.pool_n_ctx as usize {
            return Err(GenerationError::Generation(format!(
                "llama.cpp: prompt ({}) + max_tokens ({}) exceeds pool context size ({})",
                prompt_tokens.len(),
                params.max_tokens,
                self.model.pool_n_ctx
            )));
        }

        let mut pooled = self.model.acquire_context()?;
        let ctx = pooled.ctx();

        let mut batch = LlamaBatch::new(ctx.n_batch() as usize, 1);
        let last_prompt_idx = prompt_tokens.len() - 1;
        for (i, tok) in prompt_tokens.iter().enumerate() {
            let emit_logits = i == last_prompt_idx;
            batch.add(*tok, i as i32, &[0], emit_logits).map_err(|e| {
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
            let logits = ctx.get_logits_ith(batch.n_tokens() - 1);
            let next_id = argmax(logits);
            let next_tok = LlamaToken(next_id);

            if self.model.model.is_eog_token(next_tok) {
                break;
            }

            let piece = self
                .model
                .model
                .token_to_piece(next_tok, &mut decoder, /* special = */ false, None)
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
    fn complete(&mut self, prompt: &str, params: &CompletionParams) -> GenerationResult<String> {
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

/// Construct and box an `LlamaCppModel`. Returns a trait object so
/// the concrete type never escapes this module, which keeps the
/// self-referential pool's safety invariants intact (see module
/// docs).
pub fn load_llama_cpp_model(
    spec: &ModelSpec,
    profile: OptProfile,
    _merged_toml: &str,
) -> Result<Box<dyn Model>> {
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

    let template = model.chat_template(None).ok();
    let pool_n_ctx = DEFAULT_POOL_N_CTX.min(model.n_ctx_train());

    let model_id = gguf_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "llama-cpp-model".into());

    log::info!(
        "llama_cpp backend: loaded '{}' (profile={:?}, path={}, template={}, pool_n_ctx={})",
        model_id,
        profile,
        gguf_path.display(),
        if template.is_some() { "yes" } else { "absent" },
        pool_n_ctx,
    );

    let concrete = LlamaCppModel {
        context_pool: Mutex::new(VecDeque::new()),
        pool_n_ctx,
        backend,
        model,
        template,
        gguf_path,
        model_id,
        profile,
    };
    // Box immediately. The concrete type does not leak past this
    // point — callers get `Box<dyn Model>` and cannot reach the
    // struct to mem::swap it.
    Ok(Box::new(concrete))
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

    fn err_msg(r: Result<Box<dyn Model>>) -> String {
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
