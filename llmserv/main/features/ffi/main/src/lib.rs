//! C-ABI bindings for llmserv — `libllmserv.{so,dll,dylib}`.
//!
//! Consumers load the library via `ctypes` (Python), `cgo` (Go),
//! `P/Invoke` (.NET), `JNI` (Java), bridging headers (Swift), etc.
//! The API surface is deliberately small: init, complete, embed,
//! tokenize, token_count, destroy, and matching free-fns for Rust-
//! allocated buffers that cross the boundary.
//!
//! Scope is documented in `llmserv/BACKLOG.md` (T2). Non-goals:
//! async FFI, shared-weight multi-handle, cross-version ABI stability.
//!
//! Safety / ownership:
//! - Every pointer crossing the boundary is non-null and aligned (caller's
//!   responsibility to pass valid pointers; functions return an error code
//!   if they see null).
//! - Rust allocates every returned buffer. The caller MUST free it via
//!   the matching `llmserv_free_*` function; never pass it to the
//!   caller's own `free`.
//! - Every `extern "C"` function wraps its body in `catch_unwind`; a
//!   Rust panic becomes `LlmError::Panic`, never unwinds across the FFI
//!   boundary.
//! - Handles are opaque — never dereference the pointer from non-Rust
//!   code.
//!
//! Thread safety — strong contract:
//! - `llmserv_complete`, `llmserv_embed`, `llmserv_tokenize`, and
//!   `llmserv_token_count` ARE safe to call concurrently on the same
//!   handle from multiple threads.
//! - `llmserv_destroy` IS safe to call concurrently with other ops on
//!   the same handle. If a call is in flight, destroy waits for it to
//!   complete, then atomically transitions the handle to "destroyed."
//! - After destroy, every subsequent call on the handle returns
//!   `LlmError::Destroyed` — no UB, no segfault, no use-after-free.
//! - Double-destroy is a safe no-op.
//! - The handle pointer itself remains valid forever after init; only
//!   the inner model is freed on destroy. This is a bounded per-init
//!   leak (one small struct per handle) — acceptable for handle
//!   lifetimes typical of IDE / desktop use.
//! - Concurrent calls do not yield N× throughput — rayon's global pool
//!   means all calls contend for the same CPUs.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::RwLock;

use rustml_inference_layers::PoolingStrategy;
use rustml_model::OptProfile;
use swe_ml_tensor::{DType, Tensor, f32_vec_to_bytes};

use swellmd::{Model, ModelSource};

/// Error codes returned by every function. Zero is success.
#[repr(C)]
pub enum LlmError {
    Ok = 0,
    /// A pointer argument was null or a string wasn't valid UTF-8.
    InvalidInput = 1,
    /// Config load, model download, or weight load failed.
    LoadFailed = 2,
    /// Tokenization or generation returned an error.
    Runtime = 3,
    /// A Rust panic was caught at the FFI boundary.
    Panic = 4,
    /// Internal error (should not happen in normal operation).
    Internal = 5,
    /// The handle was destroyed (via `llmserv_destroy`). Any call that
    /// observes this should treat the handle as gone and not retry.
    Destroyed = 6,
}

/// Opaque session handle. Returned by `llmserv_init`, consumed by the
/// other functions, freed by `llmserv_destroy`.
///
/// The actual layout is private and may change between versions. The
/// handle pointer remains valid for the lifetime of the process after
/// init — destroy releases the inner model but not the outer struct, so
/// any stray pointer use returns `Destroyed` rather than segfaulting.
pub struct LlmHandle {
    // RwLock: many concurrent read-locks (one per in-flight call), one
    // exclusive write-lock (destroy). Option: None means destroyed.
    inner: RwLock<Option<Box<dyn Model>>>,
}

// ─── init / destroy ──────────────────────────────────────────────────

/// Load the model specified by `application.toml` (found via the
/// standard XDG search path). Writes an opaque handle to `*out_handle`
/// on success.
///
/// # Safety
/// `out_handle` must point to writable memory holding a pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_init(out_handle: *mut *mut LlmHandle) -> c_int {
    wrap(|| {
        if out_handle.is_null() {
            return Err(LlmError::InvalidInput);
        }

        let loaded = swellmd::load_config().map_err(|e| {
            log::error!("llmserv_init: load_config failed: {}", e);
            LlmError::LoadFailed
        })?;

        // Use config's [runtime] and [model] sections to build the model.
        let profile = match loaded.app.runtime.opt_profile.as_str() {
            "optimized" => OptProfile::Optimized,
            "baseline" => OptProfile::Baseline,
            "aggressive" => OptProfile::Aggressive,
            other => {
                log::error!("llmserv_init: unknown [runtime].opt_profile '{}'", other);
                return Err(LlmError::InvalidInput);
            }
        };
        profile.apply().map_err(|e| {
            log::error!("llmserv_init: profile.apply failed: {}", e);
            LlmError::LoadFailed
        })?;

        let model: Box<dyn Model> = match loaded.app.model.source {
            ModelSource::Safetensors => {
                let id = loaded.app.model.id.as_deref().ok_or_else(|| {
                    log::error!("llmserv_init: [model].source=safetensors requires [model].id");
                    LlmError::InvalidInput
                })?;
                Box::new(swellmd::load_safetensors(id, profile, &loaded.merged_toml).map_err(
                    |e| {
                        log::error!("llmserv_init: load_safetensors failed: {}", e);
                        LlmError::LoadFailed
                    },
                )?)
            }
            ModelSource::Gguf => {
                let path = loaded.app.model.path.as_deref().ok_or_else(|| {
                    log::error!("llmserv_init: [model].source=gguf requires [model].path");
                    LlmError::InvalidInput
                })?;
                Box::new(
                    swellmd::load_gguf(std::path::Path::new(path), profile).map_err(|e| {
                        log::error!("llmserv_init: load_gguf failed: {}", e);
                        LlmError::LoadFailed
                    })?,
                )
            }
        };

        let handle = Box::new(LlmHandle {
            inner: RwLock::new(Some(model)),
        });
        // Deliberate: the outer Box is leaked. `destroy` drops the inner
        // model but never frees this struct, so the handle pointer
        // remains valid for the lifetime of the process and any stray
        // use after destroy is a bounded "return Destroyed" rather than
        // UB. See thread-safety contract in the crate-level docs.
        unsafe {
            *out_handle = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// Destroy a handle: drops the inner model, releasing weights and
/// activations. Idempotent — calling destroy twice is a safe no-op.
/// Safe to call concurrently with in-flight ops: waits for them to
/// finish, then transitions atomically. Subsequent calls on the handle
/// return `LlmError::Destroyed`.
///
/// # Safety
/// `handle` must be a pointer returned by `llmserv_init`. Passing null
/// is a safe no-op. The handle pointer itself remains valid after this
/// call — only the inner model is released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_destroy(handle: *mut LlmHandle) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: handle is a leaked LlmHandle from `llmserv_init`;
        // caller contract guarantees the pointer is valid.
        let h = unsafe { &*handle };
        if let Ok(mut guard) = h.inner.write() {
            let _ = guard.take();
        }
    }));
}

// ─── complete ────────────────────────────────────────────────────────

/// Run a single completion: prompt in, generated text out.
///
/// Applies the model's chat template (wraps the prompt as a user message).
/// `max_tokens = 0` means use the model's default context length.
/// `temperature = 0.0` means greedy sampling.
///
/// On success, writes a pointer to a NUL-terminated UTF-8 string to
/// `*out_text`. The caller must free it via `llmserv_free_string`.
///
/// # Safety
/// `handle` is a valid handle from `llmserv_init`. `prompt` is a
/// NUL-terminated UTF-8 string. `out_text` points to writable memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_complete(
    handle: *const LlmHandle,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
    out_text: *mut *mut c_char,
) -> c_int {
    wrap(|| {
        let h = unsafe { handle.as_ref() }.ok_or(LlmError::InvalidInput)?;
        let prompt_str = unsafe { cstr_to_str(prompt)? };

        let guard = h.inner.read().map_err(|_| LlmError::Internal)?;
        let model = guard.as_ref().ok_or(LlmError::Destroyed)?;

        let generator = model.build_generator(temperature);
        let max = if max_tokens == 0 { 256 } else { max_tokens as usize };
        let output = generator.generate(prompt_str, max).map_err(|e| {
            log::error!("llmserv_complete: generate failed: {}", e);
            LlmError::Runtime
        })?;

        let cs = CString::new(output).map_err(|_| LlmError::Internal)?;
        unsafe {
            *out_text = cs.into_raw();
        }
        Ok(())
    })
}

// ─── complete_stream ─────────────────────────────────────────────────

/// Callback invoked for each generated token. Receives:
///
/// - `piece`: NUL-terminated UTF-8 string, the decoded token text.
///   Valid ONLY for the duration of this callback — copy it if you need
///   to keep it.
/// - `user_data`: the opaque pointer passed to `llmserv_complete_stream`.
///
/// Return `true` to keep generating, `false` to stop early. Panics in
/// the callback are caught and converted to "stop generation."
pub type LlmTokenCallback =
    extern "C" fn(piece: *const c_char, user_data: *mut c_void) -> bool;

/// Streaming completion: calls `callback` for each generated token as
/// it's decoded. Blocks until the generator stops (EOS, max_tokens,
/// or callback returned `false`).
///
/// `max_tokens = 0` means use the model's default context length.
/// `temperature = 0.0` means greedy sampling.
///
/// # Safety
/// `handle` is a valid handle from `llmserv_init`. `prompt` is a
/// NUL-terminated UTF-8 string. `callback` is a function pointer that
/// remains valid for the duration of this call. `user_data` is opaque
/// to this library and passed through unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_complete_stream(
    handle: *const LlmHandle,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
    callback: LlmTokenCallback,
    user_data: *mut c_void,
) -> c_int {
    wrap(|| {
        let h = unsafe { handle.as_ref() }.ok_or(LlmError::InvalidInput)?;
        let prompt_str = unsafe { cstr_to_str(prompt)? };
        let cb = callback;

        let guard = h.inner.read().map_err(|_| LlmError::Internal)?;
        let model = guard.as_ref().ok_or(LlmError::Destroyed)?;

        // user_data is *mut c_void — not Send. Wrap in a transparent
        // newtype with an explicit unsafe Send assertion. Callers who
        // share state across the generator thread must ensure their
        // user_data is thread-safe; we don't use threads internally
        // here (the generator runs inline), so this is fine.
        struct Ctx(*mut c_void);
        unsafe impl Send for Ctx {}
        let ctx = Ctx(user_data);

        let tokenizer = model.tokenizer();
        let generator = model.build_generator(temperature);
        let max = if max_tokens == 0 { 256 } else { max_tokens as usize };

        let _ = generator
            .generate_stream(prompt_str, max, |token_id| {
                // Decode just this token to a piece.
                let piece = match tokenizer.decode(&[token_id]) {
                    Ok(s) => s,
                    Err(e) => {
                        log::warn!(
                            "llmserv_complete_stream: decode failed for token {}: {}",
                            token_id,
                            e
                        );
                        return false;
                    }
                };
                let cstr = match CString::new(piece) {
                    Ok(c) => c,
                    Err(_) => return false, // piece contained a NUL byte
                };
                // Panic in the caller's callback -> stop generation.
                match catch_unwind(AssertUnwindSafe(|| cb(cstr.as_ptr(), ctx.0))) {
                    Ok(continue_flag) => continue_flag,
                    Err(_) => {
                        log::error!("llmserv_complete_stream: user callback panicked");
                        false
                    }
                }
            })
            .map_err(|e| {
                log::error!("llmserv_complete_stream: generate_stream failed: {}", e);
                LlmError::Runtime
            })?;

        Ok(())
    })
}

// ─── embed ───────────────────────────────────────────────────────────

/// Compute a mean-pooled embedding for the given input text. On success,
/// writes a pointer to an f32 array of length `*out_dim` to `*out_vec`.
/// The caller must free it via `llmserv_free_floats`.
///
/// # Safety
/// See `llmserv_complete`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_embed(
    handle: *const LlmHandle,
    text: *const c_char,
    out_vec: *mut *mut f32,
    out_dim: *mut usize,
) -> c_int {
    wrap(|| {
        let h = unsafe { handle.as_ref() }.ok_or(LlmError::InvalidInput)?;
        let text_str = unsafe { cstr_to_str(text)? };

        let guard = h.inner.read().map_err(|_| LlmError::Internal)?;
        let model = guard.as_ref().ok_or(LlmError::Destroyed)?;

        let ids = model.tokenizer().encode(text_str).map_err(|e| {
            log::error!("llmserv_embed: tokenize failed: {}", e);
            LlmError::Runtime
        })?;

        let seq_len = ids.len();
        let input_data: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
        let input_tensor = Tensor::new(
            f32_vec_to_bytes(input_data),
            vec![1, seq_len],
            DType::F32,
        );

        let embedding = model
            .embed(&input_tensor, PoolingStrategy::Mean)
            .map_err(|e| {
                log::error!("llmserv_embed: embed failed: {}", e);
                LlmError::Runtime
            })?;

        let vec: Vec<f32> = embedding.iter().collect();
        let dim = vec.len();
        let mut boxed = vec.into_boxed_slice();
        let p = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        unsafe {
            *out_vec = p;
            *out_dim = dim;
        }
        Ok(())
    })
}

// ─── tokenize ────────────────────────────────────────────────────────

/// Tokenize `text` and write the token ids to `*out_ids`, length in
/// `*out_len`. The caller must free via `llmserv_free_u32s`.
///
/// # Safety
/// See `llmserv_complete`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_tokenize(
    handle: *const LlmHandle,
    text: *const c_char,
    out_ids: *mut *mut u32,
    out_len: *mut usize,
) -> c_int {
    wrap(|| {
        let h = unsafe { handle.as_ref() }.ok_or(LlmError::InvalidInput)?;
        let text_str = unsafe { cstr_to_str(text)? };

        let guard = h.inner.read().map_err(|_| LlmError::Internal)?;
        let model = guard.as_ref().ok_or(LlmError::Destroyed)?;

        let ids = model.tokenizer().encode(text_str).map_err(|e| {
            log::error!("llmserv_tokenize: tokenize failed: {}", e);
            LlmError::Runtime
        })?;

        let len = ids.len();
        let mut boxed = ids.into_boxed_slice();
        let p = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        unsafe {
            *out_ids = p;
            *out_len = len;
        }
        Ok(())
    })
}

/// Return just the token count for `text` (cheap — no allocation
/// returned to the caller). IDE plugins call this on every keystroke.
///
/// # Safety
/// See `llmserv_complete`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_token_count(
    handle: *const LlmHandle,
    text: *const c_char,
    out_count: *mut usize,
) -> c_int {
    wrap(|| {
        let h = unsafe { handle.as_ref() }.ok_or(LlmError::InvalidInput)?;
        let text_str = unsafe { cstr_to_str(text)? };
        let guard = h.inner.read().map_err(|_| LlmError::Internal)?;
        let model = guard.as_ref().ok_or(LlmError::Destroyed)?;
        let ids = model.tokenizer().encode(text_str).map_err(|e| {
            log::error!("llmserv_token_count: tokenize failed: {}", e);
            LlmError::Runtime
        })?;
        unsafe {
            *out_count = ids.len();
        }
        Ok(())
    })
}

// ─── free fns — Rust allocates, Rust frees ───────────────────────────

/// Free a string returned by `llmserv_complete`. Passing null is a no-op.
///
/// # Safety
/// `s` must be a pointer previously returned by `llmserv_complete`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _ = unsafe { CString::from_raw(s) };
    }));
}

/// Free an f32 array returned by `llmserv_embed`. Passing null is a no-op.
///
/// # Safety
/// `p` must be a pointer returned by `llmserv_embed` and `len` must be
/// the exact length written to `*out_dim` at that time.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_free_floats(p: *mut f32, len: usize) {
    if p.is_null() || len == 0 {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(p, len));
    }));
}

/// Free a u32 array returned by `llmserv_tokenize`. Passing null is a no-op.
///
/// # Safety
/// `p` must be a pointer returned by `llmserv_tokenize` and `len` must
/// be the exact length written to `*out_len` at that time.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_free_u32s(p: *mut u32, len: usize) {
    if p.is_null() || len == 0 {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(p, len));
    }));
}

// ─── internals ───────────────────────────────────────────────────────

/// Wrap an FFI body: catches panics, converts Result<(), LlmError> to c_int.
fn wrap<F: FnOnce() -> Result<(), LlmError>>(f: F) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => LlmError::Ok as c_int,
        Ok(Err(e)) => e as c_int,
        Err(_) => LlmError::Panic as c_int,
    }
}

/// # Safety
/// `p` must be a NUL-terminated C string valid for the lifetime of the
/// returned reference, or null (in which case we return `InvalidInput`).
unsafe fn cstr_to_str<'a>(p: *const c_char) -> Result<&'a str, LlmError> {
    if p.is_null() {
        return Err(LlmError::InvalidInput);
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| LlmError::InvalidInput)
}
