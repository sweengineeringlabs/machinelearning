use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;

/// Backward op for the SiLU (Swish) activation.
/// saved[0] = input (pre-activation)
/// Gradient: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
pub struct SiLUBackward;

impl BackwardOp for SiLUBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let x_data = input.to_vec();
        let grad_data = grad_output.to_vec();

        let grad_input: Vec<f32> = x_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let sig = 1.0 / (1.0 + (-x).exp());
                let d_silu = sig * (1.0 + x * (1.0 - sig));
                g * d_silu
            })
            .collect();

        let result =
            Tensor::from_vec(grad_input, input.shape().to_vec()).expect("silu backward from_vec");
        vec![result]
    }

    fn name(&self) -> &str {
        "SiLUBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: SiLUBackward::backward
    #[test]
    fn test_silu_backward_output_shape_matches_input() {
        let op = SiLUBackward;
        let input = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]).unwrap();
        let grad = Tensor::ones(vec![3]);
        let grads = op.backward(&grad, &[input]);
        assert_eq!(grads[0].shape(), &[3]);
    }
}
