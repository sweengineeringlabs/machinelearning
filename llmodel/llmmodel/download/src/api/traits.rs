use crate::api::error::DownloadResult;
use crate::api::types::{GgufBundle, ModelFiles};
use std::path::Path;

/// Model download contract.
///
/// Implementations resolve a model identifier to a set of local file paths.
/// The trait is split into sync-only methods to stay usable from both
/// blocking CLIs and async servers (servers wrap in `spawn_blocking`).
pub trait Download {
    /// Download a SafeTensors-backed model synchronously. Returns paths
    /// to the cached files.
    fn download_model(&self, model_id: &str) -> DownloadResult<ModelFiles>;

    /// Download a single GGUF file from a repo synchronously.
    fn download_gguf(&self, model_id: &str, filename: &str) -> DownloadResult<GgufBundle>;

    /// Check if a model is cached locally (no network call).
    fn is_cached(&self, model_id: &str) -> bool;

    /// Return cached model files without downloading, if present.
    fn get_cached(&self, model_id: &str) -> Option<ModelFiles>;

    /// Root of the local cache directory.
    fn cache_dir(&self) -> &Path;
}
