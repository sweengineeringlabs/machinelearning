//! Error types for tensor operations

use thiserror::Error;

/// Result type alias for tensor operations
pub type TensorResult<T> = Result<T, TensorError>;

/// Errors that can occur during tensor operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Shape mismatch during operation
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Invalid dimension index
    #[error("Invalid dimension: {dim} for tensor with {ndim} dimensions")]
    InvalidDimension { dim: i64, ndim: usize },

    /// Invalid index access
    #[error("Index {index} out of bounds for dimension {dim} with size {size}")]
    IndexOutOfBounds { dim: usize, index: usize, size: usize },

    /// Invalid slice range
    #[error("Invalid slice range [{start}:{end}] for dimension with size {size}")]
    InvalidSliceRange { start: usize, end: usize, size: usize },

    /// Broadcasting error
    #[error("Cannot broadcast shapes {shape1:?} and {shape2:?}")]
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },

    /// Matrix multiplication dimension mismatch
    #[error("Matrix multiplication requires inner dimensions to match: {left} vs {right}")]
    MatmulDimensionMismatch { left: usize, right: usize },

    /// Empty tensor error
    #[error("Cannot perform operation on empty tensor")]
    EmptyTensor,

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Data conversion error
    #[error("Data conversion error: {0}")]
    ConversionError(String),

    /// DType mismatch
    #[error("DType mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch {
        expected: crate::api::dtype::DType,
        got: crate::api::dtype::DType,
    },

    /// Not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::dtype::DType;

    #[test]
    fn test_shape_mismatch_error_display_shows_shapes() {
        let err = TensorError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![4, 5],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("[2, 3]"), "expected shape missing: {}", msg);
        assert!(msg.contains("[4, 5]"), "got shape missing: {}", msg);
    }

    #[test]
    fn test_invalid_dimension_error_includes_dim_and_ndim() {
        let err = TensorError::InvalidDimension { dim: -5, ndim: 3 };
        let msg = format!("{}", err);
        assert!(msg.contains("-5"), "dim missing: {}", msg);
        assert!(msg.contains("3"), "ndim missing: {}", msg);
    }

    #[test]
    fn test_dtype_mismatch_error_includes_both_types() {
        let err = TensorError::DTypeMismatch {
            expected: DType::F32,
            got: DType::F16,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("F32"), "expected dtype missing: {}", msg);
        assert!(msg.contains("F16"), "got dtype missing: {}", msg);
    }

    #[test]
    fn test_empty_tensor_error_is_descriptive() {
        let err = TensorError::EmptyTensor;
        let msg = format!("{}", err);
        assert!(msg.contains("empty"), "message not descriptive: {}", msg);
    }

    #[test]
    fn test_tensor_result_ok_unwraps() {
        let r: TensorResult<i32> = Ok(42);
        assert_eq!(r.unwrap(), 42);
    }

    #[test]
    fn test_tensor_result_err_propagates() {
        let r: TensorResult<i32> = Err(TensorError::EmptyTensor);
        assert!(r.is_err());
    }
}
