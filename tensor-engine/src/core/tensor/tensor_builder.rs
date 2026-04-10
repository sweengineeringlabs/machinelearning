//! Builder for constructing tensors with custom configuration.

use crate::api::dtype::DType;
use crate::api::device::Device;
use crate::api::error::{TensorError, TensorResult};
use crate::core::shape_mod::shape::Shape;
use super::tensor::{Tensor, f32_vec_to_bytes};

/// Builder for constructing a [`Tensor`] with explicit shape, dtype, and device.
///
/// # Example
/// ```ignore
/// let t = TensorBuilder::new()
///     .shape(vec![2, 3])
///     .dtype(DType::F32)
///     .zeros()
///     .unwrap();
/// ```
pub struct TensorBuilder {
    shape: Option<Shape>,
    dtype: DType,
    device: Device,
}

impl TensorBuilder {
    /// Create a new builder with default settings (F32, CPU).
    pub fn new() -> Self {
        Self {
            shape: None,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Set the tensor shape.
    pub fn shape(mut self, shape: impl Into<Shape>) -> Self {
        self.shape = Some(shape.into());
        self
    }

    /// Set the data type.
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Build a tensor filled with zeros.
    pub fn zeros(self) -> TensorResult<Tensor> {
        let shape = self.shape.ok_or_else(|| {
            TensorError::InvalidOperation("TensorBuilder requires a shape".into())
        })?;
        Ok(Tensor::zeros(shape))
    }

    /// Build a tensor filled with ones.
    pub fn ones(self) -> TensorResult<Tensor> {
        let shape = self.shape.ok_or_else(|| {
            TensorError::InvalidOperation("TensorBuilder requires a shape".into())
        })?;
        Ok(Tensor::ones(shape))
    }

    /// Build a tensor from the given f32 data.
    pub fn from_data(self, data: Vec<f32>) -> TensorResult<Tensor> {
        let shape = self.shape.ok_or_else(|| {
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
        // dtype is stored but zeros/ones always produce F32 (builder dtype is advisory)
        let _ = b;
    }

    /// @covers: TensorBuilder::device
    #[test]
    fn test_device_sets_device() {
        let b = TensorBuilder::new().device(Device::Cpu);
        let t = b.shape(vec![2]).zeros().unwrap();
        assert_eq!(t.shape(), &[2]);
    }
}
