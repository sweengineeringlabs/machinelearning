use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;

/// Backward for Sigmoid: grad_output * sigmoid_output * (1 - sigmoid_output)  (FR-213)
/// saved[0] = sigmoid_output (the output of the forward pass, NOT the input)
pub(crate) struct SigmoidBackward;

impl BackwardOp for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let sig = &saved[0];
        // (1 - sigmoid_output)
        let ones = Tensor::ones(sig.shape().to_vec());
        let one_minus_sig = ones.sub_raw(sig).expect("sigmoid 1 - s");
        // sigmoid_output * (1 - sigmoid_output)
        let local_grad = sig.mul_raw(&one_minus_sig).expect("sigmoid s*(1-s)");
        // grad_output * local_grad
        let grad_input = grad_output.mul_raw(&local_grad).expect("sigmoid backward mul");
        vec![grad_input]
    }

    fn name(&self) -> &str {
        "SigmoidBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: SigmoidBackward::backward
    #[test]
    fn test_sigmoid_backward_at_half_gives_quarter() {
        let op = SigmoidBackward;
        // sigmoid(0) = 0.5; derivative at sigmoid output 0.5 is 0.5 * 0.5 = 0.25
        let sig_output = Tensor::from_vec(vec![0.5], vec![1]).unwrap();
        let grad_output = Tensor::ones(vec![1]);
        let grads = op.backward(&grad_output, &[sig_output]);
        assert!((grads[0].to_vec()[0] - 0.25).abs() < 1e-6);
    }
}
