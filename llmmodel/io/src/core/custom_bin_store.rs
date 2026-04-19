use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use swe_ml_tensor::{DType, Tensor};

use crate::api::error::{IoError, IoResult};
use crate::api::traits::{LoadTensors, SaveTensors};

/// Custom binary tensor format with trailing CRC32.
///
/// Layout: `[NumTensors: u32]`
/// Per tensor: `[NameLen: u32, Name: bytes, DType: u8, NDim: u32,
///               Shape: [u32; NDim], DataLen: u64, Data: bytes]`
/// Trailer: `[CRC32: u32]` over all preceding bytes.
#[derive(Debug, Default, Clone, Copy)]
pub struct CustomBinStore;

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

fn byte_to_dtype(b: u8) -> Result<DType, IoError> {
    match b {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        3 => Ok(DType::I8),
        4 => Ok(DType::U8),
        5 => Ok(DType::Q8_0),
        6 => Ok(DType::Q4_0),
        7 => Ok(DType::Q4_1),
        _ => Err(IoError::UnsupportedDtype(format!("Unknown dtype byte: {}", b))),
    }
}

impl SaveTensors for CustomBinStore {
    fn save(&self, path: &Path, tensors: &HashMap<String, Tensor>) -> IoResult<()> {
        let mut buf: Vec<u8> = Vec::new();

        buf.extend_from_slice(&(tensors.len() as u32).to_le_bytes());

        for (name, tensor) in tensors {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(name_bytes);

            buf.push(dtype_to_byte(tensor.dtype()));

            let shape = tensor.shape();
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());

            for &dim in shape {
                buf.extend_from_slice(&(dim as u32).to_le_bytes());
            }

            let data = tensor.as_raw_bytes()?;
            buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
            buf.extend_from_slice(data);
        }

        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        let mut file = File::create(path)?;
        file.write_all(&buf)?;
        Ok(())
    }
}

impl LoadTensors for CustomBinStore {
    fn load(&self, path: &Path) -> IoResult<HashMap<String, Tensor>> {
        let metadata = std::fs::metadata(path)?;
        if metadata.len() < 8 {
            return Err(IoError::DataCorruption("File too small".into()));
        }

        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        if buffer.len() < 4 {
            return Err(IoError::DataCorruption("File too small for CRC32".into()));
        }
        let payload = &buffer[..buffer.len() - 4];
        let stored_crc = u32::from_le_bytes(buffer[buffer.len() - 4..].try_into().unwrap());
        let computed_crc = crc32fast::hash(payload);
        if stored_crc != computed_crc {
            return Err(IoError::DataCorruption(format!(
                "CRC32 mismatch: expected {:08x}, got {:08x}",
                stored_crc, computed_crc
            )));
        }

        let mut cursor = 0;
        let num_tensors =
            u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;

        let mut tensors = HashMap::new();

        for _ in 0..num_tensors {
            if cursor + 4 > payload.len() {
                return Err(IoError::DataCorruption("Unexpected EOF".into()));
            }
            let name_len =
                u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;

            if cursor + name_len > payload.len() {
                return Err(IoError::DataCorruption("Unexpected EOF".into()));
            }
            let name = String::from_utf8(payload[cursor..cursor + name_len].to_vec())
                .map_err(|e| IoError::DataCorruption(e.to_string()))?;
            cursor += name_len;

            if cursor + 1 > payload.len() {
                return Err(IoError::DataCorruption("Unexpected EOF".into()));
            }
            let dtype = byte_to_dtype(payload[cursor])?;
            cursor += 1;

            if cursor + 4 > payload.len() {
                return Err(IoError::DataCorruption("Unexpected EOF".into()));
            }
            let ndim = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;

            let mut shape = Vec::new();
            for _ in 0..ndim {
                if cursor + 4 > payload.len() {
                    return Err(IoError::DataCorruption("Unexpected EOF".into()));
                }
                let dim =
                    u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4;
                shape.push(dim);
            }

            if cursor + 8 > payload.len() {
                return Err(IoError::DataCorruption("Unexpected EOF".into()));
            }
            let data_len =
                u64::from_le_bytes(payload[cursor..cursor + 8].try_into().unwrap()) as usize;
            cursor += 8;

            if cursor + data_len > payload.len() {
                return Err(IoError::DataCorruption("Unexpected EOF".into()));
            }
            let data = payload[cursor..cursor + data_len].to_vec();
            cursor += data_len;

            let tensor = Tensor::new(data, shape, dtype);
            tensors.insert(name, tensor);
        }

        Ok(tensors)
    }
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

        let dir = std::env::temp_dir().join("llmmodel_io_test_custom_bin");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.bin");

        let store = CustomBinStore;
        store.save(&path, &tensors).unwrap();
        let loaded = store.load(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key("weight"));
        assert!(loaded.contains_key("bias"));

        let w = &loaded["weight"];
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.dtype(), DType::F32);

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

    #[test]
    fn test_load_rejects_corrupted_crc() {
        let dir = std::env::temp_dir().join("llmmodel_io_test_crc");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.bin");

        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap(),
        );
        let store = CustomBinStore;
        store.save(&path, &tensors).unwrap();

        // Flip one byte in the payload (not in the CRC trailer) to break the checksum.
        let mut bytes = std::fs::read(&path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let result = store.load(&path);
        assert!(matches!(result, Err(IoError::DataCorruption(_))));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
