pub use crate::api::config::AppConfig;
pub use crate::api::error::*;
pub use crate::api::throttle::{Permit, Throttle};
pub use crate::api::types::*;
pub use crate::core::config::load as load_config;
pub use crate::core::loader::NativeRustBackendLoader;
pub use crate::core::router::build_router;
pub use crate::core::state::{AppState, DefaultModel};
pub use crate::core::throttle::SemaphoreThrottle;

// Shared systemd helpers — re-exported so the binary doesn't need to
// depend on swe-systemd directly.
pub use swe_systemd::{LoadedConfig, apply_logging_filter, serve_http};

// Re-export the backend-contract traits and config enums so existing
// `use swe_inference_systemd::{Model, ModelBackend, ...};` keeps working for downstream
// binaries (serve.rs, the FFI crate) without requiring them to also
// depend on `swe_inference_backend_api` directly.
pub use swe_inference_backend_api::{Model, ModelBackend, ModelBackendLoader, ModelSource, ModelSpec};
