//! `LlamaCppModel` + `LlamaCppTextCompleter`.
//!
//! The C/C++ boundary is crossed only through `llama-cpp-2`'s safe
//! wrappers. The one `unsafe` in this file is the `Send` impl on the
//! context pool — justified inline and narrow.
//!
//! Lifecycle:
//!
//! * `LlamaBackend::init()` runs exactly once per process, guarded by
//!   a `OnceLock` with a double-checked lock.
//! * `LlamaModel` (weights) is loaded once at daemon startup and lives
//!   in a pinned `self_cell` alongside the context pool. `self_cell`'s
//!   pinning prevents the self-referential-struct hazards that an
//!   unsafe transmute would leave to field-drop-order convention.
//! * **Context pool.** Sized lazily to the daemon's admission-control
//!   capacity. Requests acquire a context, decode, and return it via
//!   `clear_kv_cache_seq` + push-back. Replaces the fresh-per-call
//!   strategy that caused the tail-latency spike documented in
//!   `docs/perf/llama-cpp-vs-native.md`.
//!
//! Sampling is greedy (argmax over logits). Honors `CompletionParams
//! { temperature: 0.0 }` exactly; temp > 0 / top-k / top-p TODOs will
//! route through `rustml-generation`'s sampler so both backends share
//! sampling semantics.

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
use self_cell::self_cell;

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

/// Dependent type for the self-referential `ModelBundle`. Holds the
/// context pool with its lifetime bound to the `LlamaModel` owner.
///
/// `ContextPool` is not `Send` by default (contains `LlamaContext`
/// which wraps a raw pointer). Adding `unsafe impl Send` is sound
/// because:
///
/// 1. The `Mutex` serializes every access to the `VecDeque` and the
///    contexts inside. Two threads cannot mutate the same context
///    concurrently.
/// 2. `llama.cpp` contexts are designed for sequential use from any
///    thread. The supported usage pattern is: thread A locks, uses,
///    releases; thread B locks, uses, releases. Exactly what Mutex
///    gives us.
///
/// The corresponding `LlamaCppModel: Send + Sync` impl at the
/// bottom of this file relies on this assertion being correct.
pub struct ContextPool<'model> {
    pool: Mutex<VecDeque<LlamaContext<'model>>>,
}

// SAFETY: see `ContextPool` docs.
unsafe impl Send for ContextPool<'_> {}

self_cell!(
    /// Pinned pairing of `LlamaModel` + `ContextPool` that borrows
    /// from it. `self_cell` prevents moves that would invalidate the
    /// self-reference — unlike a manual unsafe transmute + field
    /// drop-order, a future contributor cannot accidentally break
    /// the invariant via `mem::swap` or similar.
    struct ModelBundle {
        owner: LlamaModel,
        // `not_covariant` because `ContextPool` holds `Mutex<...>`
        // which is invariant in its type parameter. This is fine —
        // we only access the dependent through the closure API
        // (`with_dependent`), which doesn't need covariance.
        #[not_covariant]
        dependent: ContextPool,
    }
);

/// A loaded llama.cpp model. Single entry point for the daemon.
pub struct LlamaCppModel {
    /// Pinned bundle holding the `LlamaModel` + its context pool.
    bundle: ModelBundle,
    backend: &'static LlamaBackend,
    template: Option<LlamaChatTemplate>,
    pool_n_ctx: u32,
    gguf_path: PathBuf,
    model_id: String,
    #[allow(dead_code)]
    profile: OptProfile,
}

// SAFETY: `ModelBundle` contains `LlamaModel` (explicitly Send+Sync
// in llama-cpp-2) and `ContextPool` (Send via our impl, Sync because
// all access is behind a Mutex). `LlamaCppModel`'s other fields are
// trivially thread-safe. We need `Sync` so `Arc<dyn Model>` in the
// daemon's `AppState` works across tokio request futures.
unsafe impl Sync for LlamaCppModel {}

impl LlamaCppModel {
    pub fn gguf_path(&self) -> &PathBuf {
        &self.gguf_path
    }

    /// Lend a context from the pool for the duration of `f`. Lazily
    /// creates a context on first acquire or when the pool is empty;
    /// clears the KV cache before `f` runs so every call starts from
    /// a blank slate; returns the context to the pool after `f`
    /// regardless of whether `f` succeeded.
    fn with_context<F, R>(&self, f: F) -> GenerationResult<R>
    where
        F: FnOnce(&LlamaModel, &mut LlamaContext<'_>) -> GenerationResult<R>,
    {
        let backend = self.backend;
        let pool_n_ctx = self.pool_n_ctx;
        self.bundle.with_dependent(|model, pool_wrapper| {
            // Acquire.
            let mut ctx = {
                let mut guard = pool_wrapper.pool.lock().map_err(|_| {
                    GenerationError::Generation("llama_cpp pool mutex poisoned".into())
                })?;
                match guard.pop_front() {
                    Some(c) => c,
                    None => {
                        drop(guard);
                        let params = LlamaContextParams::default()
                            .with_n_ctx(NonZeroU32::new(pool_n_ctx));
                        model.new_context(backend, params).map_err(|e| {
                            GenerationError::Generation(format!(
                                "llama.cpp new_context: {}",
                                e
                            ))
                        })?
                    }
                }
            };

            // Run user's closure. The context has either been
            // freshly allocated (empty KV cache) or was cleared
            // before being returned to the pool by a prior caller,
            // so it's ready for a new sequence starting at pos=0.
            let result = f(model, &mut ctx);

            // Clear the KV cache AFTER use, before returning to
            // the pool. This keeps the cheap metadata reset off
            // the request hot path — when the next request
            // acquires, the context is already clean.
            match ctx.clear_kv_cache_seq(None, None, None) {
                Ok(true) => {}
                Ok(false) => {
                    log::debug!("clear_kv_cache_seq returned false; doing full wipe");
                    ctx.clear_kv_cache();
                }
                Err(e) => {
                    log::warn!("clear_kv_cache_seq errored ({}); full wipe", e);
                    ctx.clear_kv_cache();
                }
            }

            // Return to pool. Done unconditionally so an Err from
            // `f` doesn't leak a context (the pool would lose
            // capacity permanently if we dropped on error).
            if let Ok(mut guard) = pool_wrapper.pool.lock() {
                guard.push_back(ctx);
            }

            result
        })
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
            .bundle
            .borrow_owner()
            .str_to_token(text, AddBos::Always)
            .map_err(|e| TokenizerError::TokenizerError(format!("llama.cpp tokenize: {}", e)))?;
        Ok(tokens.into_iter().map(|t| t.0 as u32).collect())
    }

    fn decode(&self, tokens: &[u32]) -> TokenizerResult<String> {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        let model = self.bundle.borrow_owner();
        for &t in tokens {
            let piece = model
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
        self.bundle.borrow_owner().n_vocab() as usize
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
}

/// Per-request text completer. Cheap to construct; all heavy state
/// lives in `LlamaCppModel`.
pub struct LlamaCppTextCompleter<'a> {
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
            .bundle
            .borrow_owner()
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

        self.model.with_context(|model, ctx| {
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

                if model.is_eog_token(next_tok) {
                    break;
                }

                let piece = model
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
        })
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
            .bundle
            .borrow_owner()
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
    let pool_n_ctx = DEFAULT_POOL_N_CTX.min(model.n_ctx_train());

    let bundle = ModelBundle::new(model, |_model| ContextPool {
        pool: Mutex::new(VecDeque::new()),
    });

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

    Ok(LlamaCppModel {
        bundle,
        backend,
        template,
        pool_n_ctx,
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
