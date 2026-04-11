use swe_ml_tensor::TensorError;
use thiserror::Error;

pub type SwetsResult<T> = Result<T, SwetsError>;

#[derive(Debug, Error)]
pub enum SwetsError {
    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),

    #[error("Tape error: {0}")]
    TapeError(String),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Training error: {0}")]
    TrainingError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_error_displays_message() {
        let err = SwetsError::TapeError("tape broken".into());
        assert!(err.to_string().contains("tape broken"));
    }

    #[test]
    fn test_shape_mismatch_error_shows_expected_and_got() {
        let err = SwetsError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![4, 5],
        };
        let msg = err.to_string();
        assert!(msg.contains("[2, 3]"));
        assert!(msg.contains("[4, 5]"));
    }

    #[test]
    fn test_invalid_config_error_displays_reason() {
        let err = SwetsError::InvalidConfig("bad lr".into());
        assert!(err.to_string().contains("bad lr"));
    }

    #[test]
    fn test_swets_result_ok_variant() {
        let result: SwetsResult<i32> = Ok(42);
        assert_eq!(result.unwrap(), 42);
    }
}
