use crate::api::tensor::Tensor;
use swe_ml_training::tape::BackwardOp;
use swe_ml_training::unbroadcast;

/// Backward for C = A - B (with broadcasting support).
/// grad_A = unbroadcast(grad_output, shape_A)
/// grad_B = unbroadcast(-grad_output, shape_B)
pub(super) struct SubBackward {
    pub(super) a_shape: Vec<usize>,
    pub(super) b_shape: Vec<usize>,
}

impl BackwardOp for SubBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        let grad_a = unbroadcast(grad_output, &self.a_shape);
        let neg_grad = grad_output.neg_raw();
        let grad_b = unbroadcast(&neg_grad, &self.b_shape);
        vec![grad_a, grad_b]
    }

    fn name(&self) -> &str {
        "NBeatsSubBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_backward_produces_correct_gradients() {
        let sb = SubBackward {
            a_shape: vec![1, 3],
            b_shape: vec![1, 3],
        };
        let grad_output = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let grads = sb.backward(&grad_output, &[]);
        assert_eq!(grads.len(), 2);
        // grad_a should equal grad_output
        assert_eq!(grads[0].to_vec(), vec![1.0, 2.0, 3.0]);
        // grad_b should be negated
        assert_eq!(grads[1].to_vec(), vec![-1.0, -2.0, -3.0]);
    }
}
