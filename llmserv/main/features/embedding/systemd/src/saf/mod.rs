pub use crate::api::config::AppConfig;
pub use crate::api::error::EmbeddingApiError;
pub use crate::core::config::load as load_config;
pub use crate::core::loader::load_gguf;
pub use crate::core::router::build_embedding_router;
pub use crate::core::state::EmbeddingState;

// Shared systemd helpers — re-exported so the binary doesn't need to
// depend on swe-systemd directly.
pub use swe_systemd::{LoadedConfig, apply_logging_filter, serve_http};
