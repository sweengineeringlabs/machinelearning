use crate::api::traits::WeightLoader;
use std::collections::HashMap;
use std::path::Path;
use swe_ml_tensor::Tensor;

/// Eager weight loader — reads all weights into RAM upfront.
///
/// This is the current default. Simple and fast for small models,
/// but uses more memory for large models.
pub struct EagerLoader;

impl WeightLoader for EagerLoader {
    fn load(&self, _path: &Path) -> Result<HashMap<String, Tensor>, String> {
        // The actual loading is done by rustml-gguf (GGUFFile::load_and_remap)
        // or rustml-hub (HubBundle::load_tensors). This provider wraps that.
        Err("EagerLoader delegates to format-specific loaders (GGUF, SafeTensors)".into())
    }

    fn describe(&self) -> &str {
        "Eager loading (all weights into RAM)"
    }
}

/// Memory-mapped weight loader — uses mmap for on-demand page loading.
///
/// The OS manages which pages are in RAM. Reduces startup time
/// and memory pressure for large models.
pub struct MmapLoader;

impl WeightLoader for MmapLoader {
    fn load(&self, _path: &Path) -> Result<HashMap<String, Tensor>, String> {
        Err("MmapLoader not yet implemented — requires GGUF mmap support".into())
    }

    fn describe(&self) -> &str {
        "Memory-mapped loading (mmap, on-demand paging)"
    }
}
