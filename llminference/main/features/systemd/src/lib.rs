//! # swe-systemd
//!
//! Shared infrastructure for long-running HTTP daemons that live under a
//! feature's `systemd/` directory. Handles the concerns every daemon
//! repeats: XDG config overlay, logging-level setup before
//! `env_logger::init()`, and the axum serve loop.

pub mod api;
mod core;
mod saf;

pub use saf::*;
