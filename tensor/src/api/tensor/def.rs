//! Core tensor type definition.
//!
//! The Tensor struct is the primary data type.
//! Implementation methods live in core/tensor/.

use crate::api::dtype::DType;
use crate::api::device::Device;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::api::storage_def::Storage;

/// Internal shape type: stack-allocated for <=4 dims.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: Tensor (Clone)
    #[test]
    fn test_tensor_clone_produces_independent_copy() {
        let t = crate::core::tensor::Tensor::zeros(vec![2, 3]);
        let t2 = t.clone();
        assert_eq!(t.shape_sv.as_slice(), t2.shape_sv.as_slice());
        assert_eq!(t.dtype, t2.dtype);
        assert_eq!(t.device, t2.device);
    }

    /// @covers: TensorShape
    #[test]
    fn test_tensor_shape_type_is_smallvec() {
        let sv: TensorShape = SmallVec::from_slice(&[2, 3, 4]);
        assert_eq!(sv.len(), 3);
        assert_eq!(sv[0], 2);
    }
}
