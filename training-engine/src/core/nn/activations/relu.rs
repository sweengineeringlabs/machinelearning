use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, TapeEntry};
use crate::api::tensor::Tensor;
use crate::core::ops::relu::ReLUBackward;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let output = input.relu_raw();

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(ReLUBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![input.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: ReLU::forward
    #[test]
    fn test_relu_forward_zeroes_negative_values() {
        let mut relu = ReLU::new();
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).unwrap();
        let output = relu.forward(&input).unwrap();
        let data = output.to_vec();
        assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    /// @covers: ReLU::parameters
    #[test]
    fn test_relu_has_no_parameters() {
        let relu = ReLU::new();
        assert!(relu.parameters().is_empty());
    }

    /// @covers: ReLU::new
    #[test]
    fn test_relu_default_matches_new() {
        let r1 = ReLU::new();
        let r2 = ReLU::default();
        assert_eq!(r1.parameters().len(), r2.parameters().len());
    }
}
