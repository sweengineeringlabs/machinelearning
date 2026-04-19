use crate::api::error::IoResult;
use std::collections::HashMap;
use std::path::Path;
use swe_ml_tensor::Tensor;

/// Read tensors from a file path. Implementations pick the on-disk format.
pub trait LoadTensors {
    fn load(&self, path: &Path) -> IoResult<HashMap<String, Tensor>>;
}

/// Write tensors to a file path. Not every format supports writing
/// (e.g. SafeTensors here is read-only), so this is a separate trait.
pub trait SaveTensors {
    fn save(&self, path: &Path, tensors: &HashMap<String, Tensor>) -> IoResult<()>;
}
