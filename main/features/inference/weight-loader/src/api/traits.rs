use std::collections::HashMap;
use std::path::Path;
use swe_ml_tensor::Tensor;

/// Contract for weight loading strategies.
///
/// Controls how model weights are loaded from disk into memory:
///
/// - **Eager**: Load all weights into RAM upfront (current default)
/// - **MemoryMapped**: Use mmap — OS manages page cache, weights read on demand
/// - **Lazy**: Load weights per-layer as needed, evict when unused
pub trait WeightLoader: Send + Sync {
    /// Load weights from the given path.
    ///
    /// Returns a HashMap of weight name → Tensor, ready for the model builder.
    fn load(&self, path: &Path) -> Result<HashMap<String, Tensor>, String>;

    /// Returns a human-readable description of the loading strategy.
    fn describe(&self) -> &str;
}
