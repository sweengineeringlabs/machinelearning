//! Core tensor and storage type definitions.
//!
//! These are data types that form the public interface.
//! Implementation methods live in core/tensor/.

use crate::api::dtype::DType;
use crate::api::device::Device;
use smallvec::SmallVec;
use std::fmt;
use std::sync::Arc;

/// Internal shape type: stack-allocated for ≤4 dims.
pub(crate) type TensorShape = SmallVec<[usize; 4]>;

/// A multi-dimensional array supporting multiple data types.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) data: Arc<Storage>,
    pub(crate) shape_sv: TensorShape,
    pub(crate) strides: TensorShape,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

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
