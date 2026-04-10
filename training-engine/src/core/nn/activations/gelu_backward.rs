use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;

/// Backward op for the approximate GELU activation.
/// saved[0] = input (pre-activation)
pub(crate) struct GELUBackward;

impl BackwardOp for GELUBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let x_data = input.to_vec();
        let grad_data = grad_output.to_vec();

        let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();

        let grad_input: Vec<f32> = x_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let x3 = x * x * x;
                let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
                let tanh_inner = inner.tanh();
                let sech2 = 1.0 - tanh_inner * tanh_inner;
                let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
                let d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner;
                g * d_gelu
            })
            .collect();

        let result =
            Tensor::from_vec(grad_input, input.shape().to_vec()).expect("gelu backward from_vec");
        vec![result]
    }

    fn name(&self) -> &str {
        "GELUBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: GELUBackward::backward
    #[test]
    fn test_gelu_backward_output_shape_matches_input() {
        let op = GELUBackward;
        let input = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]).unwrap();
        let grad = Tensor::ones(vec![3]);
        let grads = op.backward(&grad, &[input]);
        assert_eq!(grads[0].shape(), &[3]);
    }
}
