//! # swe-cli
//!
//! Shared CLI infrastructure for binaries across all workspaces.
//! Provides a [`Cli`] trait with a default `run()` that handles logging
//! setup, arg parsing, and dispatch — so each binary's `main()` is a
//! one-liner.
//!
//! Design: the trait is generic over the concrete `clap::Parser` type,
//! keeping subcommand schemas entirely per-binary while unifying the
//! one-off boilerplate that every entry point repeats (env_logger
//! init, RUST_LOG default, Cli::parse).

pub mod api;
mod core;
mod saf;

pub use saf::*;
