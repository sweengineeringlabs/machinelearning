//! Builder for constructing tensors with custom configuration.
//!
//! The data type is defined here in api/.
//! Implementation methods live in core/tensor/tensor_builder.rs.

use crate::api::dtype::DType;
use crate::api::device::Device;
use crate::api::shape::Shape;

/// Builder for constructing a [`Tensor`](crate::api::tensor::def::Tensor)
/// with explicit shape, dtype, and device.
///
/// # Example
/// ```ignore
/// let t = TensorBuilder::new()
///     .shape(vec![2, 3])
///     .zeros()
///     .unwrap();
/// ```
pub struct TensorBuilder {
    pub(crate) shape_val: Option<Shape>,
    pub(crate) dtype_val: DType,
    pub(crate) device_val: Device,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: TensorBuilder
    #[test]
    fn test_tensor_builder_struct_fields_default() {
        let b = TensorBuilder {
            shape_val: None,
            dtype_val: DType::F32,
            device_val: Device::Cpu,
        };
        assert!(b.shape_val.is_none());
        assert_eq!(b.dtype_val, DType::F32);
        assert_eq!(b.device_val, Device::Cpu);
    }
}
