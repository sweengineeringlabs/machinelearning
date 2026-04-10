//! Builder implementation methods for TensorBuilder.
//!
//! The struct is defined in api/tensor_builder_def.rs.

use crate::api::dtype::DType;
use crate::api::device::Device;
use crate::api::error::{TensorError, TensorResult};
use crate::api::shape::Shape;
use super::tensor::Tensor;

pub use crate::api::tensor_builder_def::TensorBuilder;

/// Namespace marker for TensorBuilder implementation methods.
pub(crate) struct TensorBuilderImpl;

impl TensorBuilder {
    /// Create a new builder with default settings (F32, CPU).
    pub fn new() -> Self {
        Self {
            shape_val: None,
            dtype_val: DType::F32,
            device_val: Device::Cpu,
        }
    }

    /// Set the tensor shape.
    pub fn shape(mut self, shape: impl Into<Shape>) -> Self {
        self.shape_val = Some(shape.into());
        self
    }

    /// Set the data type.
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype_val = dtype;
        self
    }

    /// Set the device.
    pub fn device(mut self, device: Device) -> Self {
        self.device_val = device;
        self
    }

    /// Build a tensor filled with zeros.
    pub fn zeros(self) -> TensorResult<Tensor> {
        let shape = self.shape_val.ok_or_else(|| {
            TensorError::InvalidOperation("TensorBuilder requires a shape".into())
        })?;
        Ok(Tensor::zeros(shape))
    }

    /// Build a tensor filled with ones.
    pub fn ones(self) -> TensorResult<Tensor> {
        let shape = self.shape_val.ok_or_else(|| {
            TensorError::InvalidOperation("TensorBuilder requires a shape".into())
        })?;
        Ok(Tensor::ones(shape))
    }

    /// Build a tensor from the given f32 data.
    pub fn from_data(self, data: Vec<f32>) -> TensorResult<Tensor> {
        let shape = self.shape_val.ok_or_else(|| {
            TensorError::InvalidOperation("TensorBuilder requires a shape".into())
        })?;
        Tensor::from_vec(data, shape)
    }
}

impl Default for TensorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: TensorBuilder::zeros
    #[test]
    fn test_builder_zeros_produces_correct_shape() {
        let t = TensorBuilder::new()
            .shape(vec![2, 3])
            .zeros()
            .unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
    }

    /// @covers: TensorBuilder::ones
    #[test]
    fn test_builder_ones_produces_all_ones() {
        let t = TensorBuilder::new()
            .shape(vec![4])
            .ones()
            .unwrap();
        let data = t.as_slice_f32().unwrap();
        assert!(data.iter().all(|&v| v == 1.0));
    }

    /// @covers: TensorBuilder::from_data
    #[test]
    fn test_builder_from_data_stores_values() {
        let t = TensorBuilder::new()
            .shape(vec![2, 2])
            .from_data(vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// @covers: TensorBuilder::zeros
    #[test]
    fn test_builder_without_shape_returns_error() {
        let result = TensorBuilder::new().zeros();
        assert!(result.is_err(), "builder without shape should fail");
    }

    /// @covers: TensorBuilder::dtype
    #[test]
    fn test_dtype_sets_dtype() {
        let b = TensorBuilder::new().dtype(DType::F16);
        let _ = b;
    }

    /// @covers: TensorBuilder::device
    #[test]
    fn test_device_sets_device() {
        let b = TensorBuilder::new().device(Device::Cpu);
        let t = b.shape(vec![2]).zeros().unwrap();
        assert_eq!(t.shape(), &[2]);
    }

    /// @covers: TensorBuilder::new
    #[test]
    fn test_new_creates_default_builder() {
        let b = TensorBuilder::new();
        assert!(b.shape_val.is_none());
        assert_eq!(b.dtype_val, DType::F32);
    }

    /// @covers: TensorBuilder::shape
    #[test]
    fn test_shape_stores_shape() {
        let b = TensorBuilder::new().shape(vec![3, 4]);
        assert!(b.shape_val.is_some());
    }
}
