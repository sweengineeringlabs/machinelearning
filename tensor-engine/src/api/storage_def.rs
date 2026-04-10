//! Storage type definition and byte-conversion helpers.

use std::fmt;
use std::sync::Arc;

use crate::api::tensor::def::Tensor;

/// Underlying storage for tensor data.
pub enum Storage {
    Owned(Vec<u8>),
    View {
        parent: Arc<Tensor>,
        offset: usize,
        len: usize,
    },
    MMap {
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        len: usize,
    },
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Storage::Owned(v) => write!(f, "Owned({} bytes)", v.len()),
            Storage::View { offset, len, .. } => {
                write!(f, "View(offset={}, len={})", offset, len)
            }
            Storage::MMap { offset, len, .. } => {
                write!(f, "MMap(offset={}, len={})", offset, len)
            }
        }
    }
}

/// Returns the byte length of the given storage.
pub(crate) fn storage_byte_len(s: &Storage) -> usize {
    match s {
        Storage::Owned(v) => v.len(),
        Storage::View { len, .. } => *len,
        Storage::MMap { len, .. } => *len,
    }
}

/// Safely convert a Vec<f32> into a Vec<u8> without unsafe code.
pub fn f32_vec_to_bytes(v: Vec<f32>) -> Vec<u8> {
    match bytemuck::try_cast_vec::<f32, u8>(v) {
        Ok(bytes) => bytes,
        Err((_, original)) => bytemuck::cast_slice::<f32, u8>(&original).to_vec(),
    }
}

/// Safely reinterpret an f32 slice as bytes.
pub fn f32_slice_to_bytes(s: &[f32]) -> &[u8] {
    bytemuck::cast_slice(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: storage_byte_len
    #[test]
    fn test_storage_byte_len_owned_returns_vec_length() {
        let s = Storage::Owned(vec![0u8; 100]);
        assert_eq!(storage_byte_len(&s), 100);
    }

    /// @covers: storage_byte_len
    #[test]
    fn test_storage_byte_len_owned_empty() {
        let s = Storage::Owned(vec![]);
        assert_eq!(storage_byte_len(&s), 0);
    }

    /// @covers: f32_vec_to_bytes
    #[test]
    fn test_f32_vec_to_bytes_correct_byte_count() {
        let bytes = f32_vec_to_bytes(vec![1.0, 2.0, 3.0]);
        assert_eq!(bytes.len(), 12);
    }

    /// @covers: f32_vec_to_bytes
    #[test]
    fn test_f32_vec_to_bytes_preserves_values() {
        let original = vec![42.0f32];
        let bytes = f32_vec_to_bytes(original);
        let recovered: &[f32] = bytemuck::cast_slice(&bytes);
        assert_eq!(recovered[0], 42.0);
    }

    /// @covers: f32_slice_to_bytes
    #[test]
    fn test_f32_slice_to_bytes_four_bytes_per_float() {
        let data = [1.0f32, 2.0];
        let bytes = f32_slice_to_bytes(&data);
        assert_eq!(bytes.len(), 8);
    }

    /// @covers: Storage (Debug)
    #[test]
    fn test_storage_debug_owned_shows_byte_count() {
        let s = Storage::Owned(vec![0u8; 64]);
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("64"), "debug should contain byte count: {}", dbg);
    }
}
