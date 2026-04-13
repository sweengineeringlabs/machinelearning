//! llama.cpp-backed model backend.
//!
//! Everything in this crate is gated behind the `llama-cpp` feature. When
//! the feature is off, the crate compiles to an empty shell — workspace
//! builds don't require a C++ toolchain. When on, it exposes
//! `LlamaCppBackendLoader`, a `llmbackend::ModelBackendLoader` impl that
//! loads GGUF models via the `llama-cpp-2` crate.
//!
//! The daemon gates its dependency on this crate behind the matching
//! `backend-llama-cpp` cargo feature, which forwards to this crate's
//! `llama-cpp` feature. Registration in the daemon's backend registry is
//! `#[cfg(feature = "backend-llama-cpp")]` so the `LlamaCpp` config
//! variant is only reachable when the feature is on.

#[cfg(feature = "llama-cpp")]
pub mod api;
#[cfg(feature = "llama-cpp")]
mod core;
#[cfg(feature = "llama-cpp")]
mod saf;

#[cfg(feature = "llama-cpp")]
pub use saf::*;
