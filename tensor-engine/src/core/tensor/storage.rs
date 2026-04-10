//! Underlying storage backends for tensor data.

use std::fmt;
use std::sync::Arc;

/// Underlying storage for tensor data.
pub enum Storage {
    Owned(Vec<u8>),
    View {
        parent: Arc<super::tensor::Tensor>,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: Storage::Owned
    #[test]
    fn test_storage_owned_debug_shows_byte_count() {
        let s = Storage::Owned(vec![0u8; 42]);
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("42"), "expected '42' in debug output: {}", dbg);
    }

    /// @covers: storage_byte_len
    #[test]
    fn test_storage_byte_len_owned_returns_vec_length() {
        let s = Storage::Owned(vec![1, 2, 3, 4, 5]);
        assert_eq!(storage_byte_len(&s), 5);
    }

    /// @covers: storage_byte_len
    #[test]
    fn test_storage_byte_len_owned_empty_returns_zero() {
        let s = Storage::Owned(vec![]);
        assert_eq!(storage_byte_len(&s), 0);
    }

    /// @covers: Storage::fmt
    #[test]
    fn test_fmt_view_debug_shows_offset_and_len() {
        use std::sync::Arc;
        let parent = Arc::new(crate::core::tensor::Tensor::zeros(vec![10]));
        let s = Storage::View { parent, offset: 4, len: 8 };
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("View"), "expected 'View' in: {}", dbg);
        assert!(dbg.contains("4"), "expected offset in: {}", dbg);
    }
}
