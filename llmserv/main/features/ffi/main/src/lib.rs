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

#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use rustml_inference_layers::PoolingStrategy;
use rustml_model::OptProfile;
use rustml_tokenizer::Tokenizer;
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
}

/// Opaque session handle. Returned by `llmserv_init`, consumed by the
/// other functions, freed by `llmserv_destroy`.
///
/// The actual layout is private and may change between versions.
pub struct LlmHandle {
    model: Box<dyn Model>,
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

        let handle = Box::new(LlmHandle { model });
        unsafe {
            *out_handle = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// Destroy a handle returned by `llmserv_init`. Calling any other
/// function with the handle after destroy is undefined behavior.
///
/// # Safety
/// `handle` must be a pointer returned by `llmserv_init` that has not
/// already been destroyed. Passing null is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llmserv_destroy(handle: *mut LlmHandle) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let _ = unsafe { Box::from_raw(handle) };
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
        let session = unsafe { handle.as_ref() }.ok_or(LlmError::InvalidInput)?;
        let prompt_str = unsafe { cstr_to_str(prompt)? };

        let generator = session.model.build_generator(temperature);
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
        let session = unsafe { handle.as_ref() }.ok_or(LlmError::InvalidInput)?;
        let text_str = unsafe { cstr_to_str(text)? };

        let ids = session.model.tokenizer().encode(text_str).map_err(|e| {
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

        let embedding = session
            .model
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
        let session = unsafe { handle.as_ref() }.ok_or(LlmError::InvalidInput)?;
        let text_str = unsafe { cstr_to_str(text)? };

        let ids = session.model.tokenizer().encode(text_str).map_err(|e| {
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
        let session = unsafe { handle.as_ref() }.ok_or(LlmError::InvalidInput)?;
        let text_str = unsafe { cstr_to_str(text)? };
        let ids = session.model.tokenizer().encode(text_str).map_err(|e| {
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
