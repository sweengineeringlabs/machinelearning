//! `LlamaCppModel` + `LlamaCppTextCompleter`.
//!
//! The C/C++ boundary is crossed only through the `llama-cpp-2` safe
//! wrappers — no `unsafe` in the forward-pass code. There is one
//! `unsafe impl Send` for the pooled-context wrapper, justified
//! inline where it appears.
//!
//! Lifecycle:
//!
//! * `LlamaBackend::init()` is called exactly once per process, owned
//!   by a `OnceLock` with a double-checked lock around init.
//! * `LlamaModel` (weights) is loaded once at daemon startup and held
//!   for the daemon's lifetime.
//! * **Context pool.** `LlamaCppModel` owns a
//!   `Mutex<VecDeque<OwnedContext>>` sized to the daemon's
//!   admission-control capacity. Completers acquire a context via
//!   `with_context`; the context's KV cache is cleared before use
//!   and the context is returned to the pool on scope exit. This
//!   replaces the earlier fresh-context-per-call strategy that was
//!   causing the tail latency spike documented in
//!   `docs/perf/llama-cpp-vs-native.md`.
//!
//! Sampling is greedy (argmax over logits). This honors `temperature
//! = 0.0` from `CompletionParams` exactly; temperature > 0 / top-k /
//! top-p are TODOs that will plug into the existing sampler in
//! `rustml-generation` so both backends share sampling semantics.

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

/// Process-wide `llama.cpp` backend handle. `llama_backend_init` may
/// only succeed once per process; subsequent calls error with
/// `BackendAlreadyInitialized`. We use double-checked locking so only
/// the first caller runs `init()`; later callers get the cached
/// handle.
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
/// context in the pool pre-allocates KV cache for this many tokens.
/// If a request's `prompt_tokens + max_tokens` exceeds this, the
/// request fails early with a clear error — no re-alloc at request
/// time (which would defeat the pool's purpose).
///
/// 8192 is a sensible default for 1-3B models on CPU: prompt+decode
/// for typical chat fits, and the ~100 MB KV allocation per context
/// is acceptable. Larger models should be more conservative; smaller
/// models could afford more. Making this per-model-configurable is
/// future work.
const DEFAULT_POOL_N_CTX: u32 = 8192;

/// Wraps `LlamaContext` with a transmuted `'static` lifetime so it
/// can live inside `LlamaCppModel` alongside the `LlamaModel` it
/// borrows from. This is a self-referential struct.
///
/// # Safety invariants
///
/// The `'static` lifetime is a lie — the context's C-level pointers
/// are stable independent of Rust's borrow checker, but Rust's type
/// system doesn't know that. We preserve safety via field drop order:
/// `LlamaCppModel.context_pool` is declared before `LlamaCppModel.model`,
/// so the pool (and all its `OwnedContext`s) is dropped first when
/// `LlamaCppModel` drops. No `OwnedContext` is ever accessible after
/// the `LlamaModel` it points to has been freed.
///
/// `Send` is safe because:
///   1. A given `OwnedContext` is only ever handled through a `Mutex`
///      (the context pool) or as a local in `run_decode`. Concurrent
///      mutation from two threads is impossible.
///   2. llama.cpp's context is designed for serial use from any
///      thread. Sequential use across threads is the supported
///      pattern (that's how Ollama does it).
struct OwnedContext {
    inner: LlamaContext<'static>,
}

// SAFETY: see documentation on `OwnedContext`.
unsafe impl Send for OwnedContext {}

/// Handle to a context borrowed from the pool. Returns the context
/// on `Drop`. Kept private to this crate — callers use
/// `LlamaCppModel::with_context`.
struct PooledContext<'m> {
    owner: &'m LlamaCppModel,
    ctx: Option<OwnedContext>,
}

impl PooledContext<'_> {
    /// The returned `&mut LlamaContext<'static>` carries our internal
    /// lifetime lie (see `OwnedContext` safety docs). The borrow is
    /// bounded by `&mut self`, which is bounded by `self.owner` — so
    /// callers can't outlive the underlying `LlamaModel` even though
    /// the literal lifetime parameter says `'static`.
    fn ctx(&mut self) -> &mut LlamaContext<'static> {
        &mut self.ctx.as_mut().expect("ctx present until drop").inner
    }
}

impl Drop for PooledContext<'_> {
    fn drop(&mut self) {
        if let Some(mut owned) = self.ctx.take() {
            // Clear all sequences so the next acquirer sees a blank KV
            // cache. If clearing fails (shouldn't, but be defensive),
            // drop the context instead of returning a dirty one to the
            // pool — next acquire will create a fresh one.
            match owned.inner.clear_kv_cache_seq(None, None, None) {
                Ok(_) => {
                    if let Ok(mut pool) = self.owner.context_pool.lock() {
                        pool.push_back(owned);
                    }
                    // Mutex poisoned — let the context drop here.
                }
                Err(e) => {
                    log::warn!(
                        "llama_cpp: clear_kv_cache_seq failed ({}); dropping context instead of pooling",
                        e
                    );
                }
            }
        }
    }
}

/// A loaded llama.cpp model. Holds weights for the daemon's lifetime
/// and a pool of reusable contexts.
///
/// Field order matters for safety — see `OwnedContext` docs. The pool
/// must drop BEFORE the model so transmuted `'static` lifetimes don't
/// outlive their real owner.
pub struct LlamaCppModel {
    // Dropped first: pool of contexts. Each references `model` below.
    context_pool: Mutex<VecDeque<OwnedContext>>,
    pool_n_ctx: u32,
    // Dropped last: weights. Everything above must go first.
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

    fn build_context(&self) -> Result<OwnedContext, GenerationError> {
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.pool_n_ctx));
        let ctx = self.model.new_context(self.backend, params).map_err(|e| {
            GenerationError::Generation(format!("llama.cpp new_context failed: {}", e))
        })?;
        // SAFETY: see `OwnedContext` docs. The 'static lifetime is
        // a lie bounded by self.model via field drop order.
        let static_ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
        Ok(OwnedContext { inner: static_ctx })
    }

    fn acquire_context(&self) -> Result<PooledContext<'_>, GenerationError> {
        let maybe_pooled = {
            let mut pool = self.context_pool.lock().map_err(|_| {
                GenerationError::Generation("llama_cpp context pool mutex poisoned".into())
            })?;
            pool.pop_front()
        };
        let ctx = match maybe_pooled {
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
        // `special = true` so control tokens (BOS, EOS, chat-turn
        // markers) render as their string form instead of erroring
        // with `UnknownTokenType`. Callers who want raw text without
        // special tokens should filter them beforehand — our
        // generation loop already skips EOG via `is_eog_token` so
        // streamed output is already clean.
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        for &t in tokens {
            let piece = self
                .model
                .token_to_piece(LlamaToken(t as i32), &mut decoder, true, None)
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

/// Per-request text completer. Borrows the `LlamaCppModel` so it can
/// pull contexts from its pool.
pub struct LlamaCppTextCompleter<'a> {
    model: &'a LlamaCppModel,
    template: Option<&'a LlamaChatTemplate>,
}

impl LlamaCppTextCompleter<'_> {
    /// Core decode loop. Acquires a context from the pool (or creates
    /// one if the pool is empty), tokenizes the prompt, runs the
    /// forward pass, greedy-samples, invokes the callback per token,
    /// returns the full generated text (without the prompt). The
    /// context returns to the pool on scope exit.
    fn run_decode(
        &self,
        prompt: &str,
        params: &CompletionParams,
        mut callback: Option<&mut dyn FnMut(u32) -> bool>,
    ) -> GenerationResult<String> {
        let prompt_tokens = self.model.model.str_to_token(prompt, AddBos::Always).map_err(|e| {
            GenerationError::Generation(format!("llama.cpp tokenize failed: {}", e))
        })?;

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

            if self.model.model.is_eog_token(next_tok) {
                break;
            }

            let piece = self
                .model
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

    let template = model.chat_template(None).ok();

    let model_id = gguf_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "llama-cpp-model".into());

    // Cap pool ctx size at what the model was trained for. Some small
    // models have n_ctx_train < DEFAULT_POOL_N_CTX.
    let pool_n_ctx = DEFAULT_POOL_N_CTX.min(model.n_ctx_train());

    log::info!(
        "llama_cpp backend: loaded '{}' (profile={:?}, path={}, template={}, pool_n_ctx={})",
        model_id,
        profile,
        gguf_path.display(),
        if template.is_some() { "yes" } else { "absent" },
        pool_n_ctx,
    );

    Ok(LlamaCppModel {
        context_pool: Mutex::new(VecDeque::new()),
        pool_n_ctx,
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
