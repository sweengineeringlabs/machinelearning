//! SafeTensors format loader
//!
//! SafeTensors is a simple and safe format for storing tensors.
//! Format specification: https://github.com/huggingface/safetensors
//!
//! Uses zero-copy mmap via the `safetensors` crate, preserving original dtype.

use crate::api::error::HubResult;
use swe_ml_tensor::{DType, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Errors specific to SafeTensors loading
#[derive(Error, Debug)]
pub enum SafeTensorsError {
    #[error("Invalid header: {0}")]
    InvalidHeader(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Data corruption: {0}")]
    DataCorruption(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Load SafeTensors with zero-copy mmap, preserving original dtype.
///
/// This is the preferred loading method for large models as it:
/// - Avoids allocating memory for the entire file
/// - Returns tensors backed by the memory-mapped file
/// - Preserves F16/BF16 dtypes (convert with `tensor.to_f32()` when needed)
pub fn load_safetensors(path: &Path) -> HubResult<HashMap<String, Tensor>> {
    let file = File::open(path)?;
    // SAFETY: The file is opened read-only and lives for the duration of the
    // returned tensors (via Arc<Mmap>). Memory-mapped reads are safe as long as
    // no external process truncates/modifies the file while mapped.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let mmap_arc = Arc::new(mmap);

    let st = safetensors::SafeTensors::deserialize(mmap_arc.as_ref())
        .map_err(|e| SafeTensorsError::InvalidHeader(e.to_string()))?;

    let mut tensors = HashMap::new();

    for (name, view) in st.tensors() {
        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::F16 => DType::F16,
            other => {
                return Err(SafeTensorsError::UnsupportedDtype(format!("{:?}", other)).into());
            }
        };

        let shape: Vec<usize> = view.shape().to_vec();
        let data_len = view.data().len();

        // Calculate offset relative to mmap start
        let mmap_ptr = mmap_arc.as_ptr() as usize;
        let data_ptr = view.data().as_ptr() as usize;
        let offset = data_ptr - mmap_ptr;

        let tensor = Tensor::from_mmap(mmap_arc.clone(), offset, data_len, shape, dtype);
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}

/// DType byte encoding for custom binary format
fn dtype_to_byte(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I8 => 3,
        DType::U8 => 4,
        DType::Q8_0 => 5,
        DType::Q4_0 => 6,
        DType::Q4_1 => 7,
    }
}

fn byte_to_dtype(b: u8) -> Result<DType, SafeTensorsError> {
    match b {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        3 => Ok(DType::I8),
        4 => Ok(DType::U8),
        5 => Ok(DType::Q8_0),
        6 => Ok(DType::Q4_0),
        7 => Ok(DType::Q4_1),
        _ => Err(SafeTensorsError::UnsupportedDtype(format!("Unknown dtype byte: {}", b))),
    }
}

/// Save tensors in custom binary format with trailing CRC32.
///
/// Format: \[NumTensors: u32\]
/// Per tensor: \[NameLen: u32, Name: bytes, DType: u8, NDim: u32, Shape: \[u32; NDim\], DataLen: u64, Data: bytes\]
/// Trailer: \[CRC32: u32\] (over all preceding bytes)
pub fn save_custom_bin(path: &Path, tensors: &HashMap<String, Tensor>) -> HubResult<()> {
    use std::io::Write;

    let mut buf: Vec<u8> = Vec::new();

    // NumTensors
    buf.extend_from_slice(&(tensors.len() as u32).to_le_bytes());

    for (name, tensor) in tensors {
        // NameLen + Name
        let name_bytes = name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        // DType
        buf.push(dtype_to_byte(tensor.dtype()));

        // NDim
        let shape = tensor.shape();
        buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());

        // Shape
        for &dim in shape {
            buf.extend_from_slice(&(dim as u32).to_le_bytes());
        }

        // Data
        let data = tensor.as_raw_bytes()?;
        buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        buf.extend_from_slice(data);
    }

    // CRC32 over all preceding bytes
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    let mut file = File::create(path)?;
    file.write_all(&buf)?;
    Ok(())
}

/// Load tensors from custom binary format with CRC32 verification.
pub fn load_custom_bin(path: &Path) -> HubResult<HashMap<String, Tensor>> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() < 8 {
        return Err(SafeTensorsError::DataCorruption("File too small".into()).into());
    }

    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer)?;

    // Verify CRC32: last 4 bytes are the checksum
    if buffer.len() < 4 {
        return Err(SafeTensorsError::DataCorruption("File too small for CRC32".into()).into());
    }
    let payload = &buffer[..buffer.len() - 4];
    let stored_crc = u32::from_le_bytes(buffer[buffer.len() - 4..].try_into().unwrap());
    let computed_crc = crc32fast::hash(payload);
    if stored_crc != computed_crc {
        return Err(SafeTensorsError::DataCorruption(format!(
            "CRC32 mismatch: expected {:08x}, got {:08x}",
            stored_crc, computed_crc
        ))
        .into());
    }

    let mut cursor = 0;

    let num_tensors = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
    cursor += 4;

    let mut tensors = HashMap::new();

    for _ in 0..num_tensors {
        // Name
        if cursor + 4 > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let name_len = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;

        if cursor + name_len > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let name = String::from_utf8(payload[cursor..cursor + name_len].to_vec())
            .map_err(|e| SafeTensorsError::DataCorruption(e.to_string()))?;
        cursor += name_len;

        // DType
        if cursor + 1 > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let dtype = byte_to_dtype(payload[cursor])?;
        cursor += 1;

        // NDim
        if cursor + 4 > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let ndim = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;

        // Shape
        let mut shape = Vec::new();
        for _ in 0..ndim {
            if cursor + 4 > payload.len() {
                return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
            }
            let dim = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            shape.push(dim);
        }

        // Data
        if cursor + 8 > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let data_len = u64::from_le_bytes(payload[cursor..cursor + 8].try_into().unwrap()) as usize;
        cursor += 8;

        if cursor + data_len > payload.len() {
            return Err(SafeTensorsError::DataCorruption("Unexpected EOF".into()).into());
        }
        let data = payload[cursor..cursor + data_len].to_vec();
        cursor += data_len;

        let tensor = Tensor::new(data, shape, dtype);
        tensors.insert(name, tensor);
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_bin_roundtrip() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap(),
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::from_vec(vec![0.1, 0.2], vec![2]).unwrap(),
        );

        let dir = std::env::temp_dir().join("rustml_test_custom_bin");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.bin");

        save_custom_bin(&path, &tensors).unwrap();
        let loaded = load_custom_bin(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key("weight"));
        assert!(loaded.contains_key("bias"));

        let w = &loaded["weight"];
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.dtype(), DType::F32);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_dtype_byte_roundtrip() {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::Q8_0, DType::Q4_0] {
            let b = dtype_to_byte(dtype);
            let back = byte_to_dtype(b).unwrap();
            assert_eq!(back, dtype);
        }
    }
}
