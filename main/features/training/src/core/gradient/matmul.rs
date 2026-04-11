use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;

/// Backward for C = A @ B
/// saved[0] = A, saved[1] = B
/// grad_A = grad_output @ B^T
/// grad_B = A^T @ grad_output
pub struct MatMulBackward;

impl BackwardOp for MatMulBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let a = &saved[0];
        let b = &saved[1];

        // grad_A = grad_output @ B^T
        let b_t = b.transpose_raw(-1, -2).expect("transpose B");
        let grad_a = grad_output.matmul_raw(&b_t).expect("matmul grad_a");

        // grad_B = A^T @ grad_output
        let a_t = a.transpose_raw(-1, -2).expect("transpose A");
        let grad_b = a_t.matmul_raw(grad_output).expect("matmul grad_b");

        vec![grad_a, grad_b]
    }

    fn name(&self) -> &str {
        "MatMulBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: MatMulBackward::backward
    #[test]
    fn test_matmul_backward_produces_two_gradients() {
        let op = MatMulBackward;
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![3, 4]);
        let grad_output = Tensor::ones(vec![2, 4]);
        let grads = op.backward(&grad_output, &[a, b]);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 3]); // grad_A
        assert_eq!(grads[1].shape(), &[3, 4]); // grad_B
    }
}
