use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use swe_ml_tensor::{DType, Tensor};

use crate::api::error::{IoError, IoResult};
use crate::api::traits::LoadTensors;

/// SafeTensors format reader. Uses zero-copy mmap and preserves the
/// on-disk dtype (F32 / BF16 / F16).
#[derive(Debug, Default, Clone, Copy)]
pub struct SafeTensorsStore;

impl LoadTensors for SafeTensorsStore {
    fn load(&self, path: &Path) -> IoResult<HashMap<String, Tensor>> {
        let file = File::open(path)?;
        // SAFETY: The file is opened read-only and the mmap lives for the
        // duration of the returned tensors (via Arc<Mmap>). Memory-mapped
        // reads are safe as long as no external process truncates or
        // modifies the file while mapped.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mmap_arc = Arc::new(mmap);

        let st = safetensors::SafeTensors::deserialize(mmap_arc.as_ref())
            .map_err(|e| IoError::InvalidHeader(e.to_string()))?;

        let mut tensors = HashMap::new();

        for (name, view) in st.tensors() {
            let dtype = match view.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::BF16 => DType::BF16,
                safetensors::Dtype::F16 => DType::F16,
                other => {
                    return Err(IoError::UnsupportedDtype(format!("{:?}", other)));
                }
            };

            let shape: Vec<usize> = view.shape().to_vec();
            let data_len = view.data().len();

            let mmap_ptr = mmap_arc.as_ptr() as usize;
            let data_ptr = view.data().as_ptr() as usize;
            let offset = data_ptr - mmap_ptr;

            let tensor = Tensor::from_mmap(mmap_arc.clone(), offset, data_len, shape, dtype);
            tensors.insert(name, tensor);
        }

        Ok(tensors)
    }
}
