//! Backend contract: what it means to be a "model" that the daemon can
//! serve, and how the daemon loads one.
//!
//! This crate is deliberately thin. It owns:
//!
//! - `Model` — the DI seam the HTTP router calls into.
//! - `ModelBackendLoader` — the SPI the daemon's startup code dispatches
//!    over to construct a `Model`.
//! - `ModelSpec` / `ModelBackend` / `ModelSource` — the config schema that
//!    drives loader selection.
//!
//! Backends (native-Rust, llama.cpp, remote HTTP, test mocks) depend on
//! this crate only. They do not depend on the daemon's HTTP stack.

pub mod api;
mod saf;

pub use saf::*;
