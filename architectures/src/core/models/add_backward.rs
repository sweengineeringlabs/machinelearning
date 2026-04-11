use crate::api::tensor::Tensor;
use swe_ml_training::tape::BackwardOp;
use swe_ml_training::unbroadcast;

/// Backward for C = A + B (with broadcasting support).
pub(super) struct AddBackward {
    pub(super) a_shape: Vec<usize>,
    pub(super) b_shape: Vec<usize>,
}

impl BackwardOp for AddBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        let grad_a = unbroadcast(grad_output, &self.a_shape);
        let grad_b = unbroadcast(grad_output, &self.b_shape);
        vec![grad_a, grad_b]
    }

    fn name(&self) -> &str {
        "NBeatsAddBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_backward_produces_identity_gradients() {
        let ab = AddBackward {
            a_shape: vec![1, 3],
            b_shape: vec![1, 3],
        };
        let grad_output = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let grads = ab.backward(&grad_output, &[]);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(grads[1].to_vec(), vec![1.0, 2.0, 3.0]);
    }
}
