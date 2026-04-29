//! Proto-generated types for `EmbedService`.
//!
//! Generated at build time by `build.rs` from `proto/embed.proto` and
//! included from `OUT_DIR`.  Re-exported through this module so the rest
//! of the crate refers to a stable path (`crate::api::proto::*`).

#![allow(missing_docs)]
#![allow(clippy::derive_partial_eq_without_eq)]

include!(concat!(env!("OUT_DIR"), "/justembed.rs"));

/// Fully-qualified gRPC method path for `EmbedService.Embed`.
///
/// Used by the gRPC ingress to route a request to the right handler.
/// Kept as a single source of truth so the server-side dispatch and the
/// handler registration cannot drift apart.
pub const EMBED_METHOD_PATH: &str = "/justembed.EmbedService/Embed";
