//! E2E tests for the TensorError type.

use tensor_engine::{TensorError, TensorResult};

/// @covers: TensorError::ShapeMismatch
#[test]
fn test_shape_mismatch_error_format() {
    let err = TensorError::ShapeMismatch {
        expected: vec![2, 3],
        got: vec![4, 5],
    };
    let msg = format!("{}", err);
    assert!(msg.contains("[2, 3]"));
    assert!(msg.contains("[4, 5]"));
}

/// @covers: TensorError::EmptyTensor
#[test]
fn test_empty_tensor_error_is_descriptive() {
    let err = TensorError::EmptyTensor;
    let msg = format!("{}", err);
    assert!(msg.contains("empty"));
}

/// @covers: TensorResult
#[test]
fn test_tensor_result_ok_unwraps() {
    let r: TensorResult<i32> = Ok(42);
    assert_eq!(r.unwrap(), 42);
}

/// @covers: TensorResult
#[test]
fn test_tensor_result_err_propagates() {
    let r: TensorResult<i32> = Err(TensorError::EmptyTensor);
    assert!(r.is_err());
}
