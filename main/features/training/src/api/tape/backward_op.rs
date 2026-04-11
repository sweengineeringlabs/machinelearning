use crate::api::tensor::Tensor;

// --- BackwardOp trait ---

pub trait BackwardOp: Send + Sync {
    /// Compute gradients for inputs given the gradient of the output.
    /// Returns one gradient per input_id (in the same order as input_ids in TapeEntry).
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor>;
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopBackward;

    impl BackwardOp for NoopBackward {
        fn backward(&self, _grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
            vec![]
        }
        fn name(&self) -> &str {
            "NoopBackward"
        }
    }

    #[test]
    fn test_backward_op_name_returns_identifier() {
        let op = NoopBackward;
        assert_eq!(op.name(), "NoopBackward");
    }

    #[test]
    fn test_backward_op_backward_returns_empty_grads_for_noop() {
        let op = NoopBackward;
        let grad = Tensor::ones(vec![2, 3]);
        let result = op.backward(&grad, &[]);
        assert!(result.is_empty());
    }
}
