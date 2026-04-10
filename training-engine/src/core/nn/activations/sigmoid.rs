use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, TapeEntry};
use crate::api::tensor::Tensor;
use crate::core::ops::sigmoid::SigmoidBackward;

pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let output = Tensor::new(input.inner().sigmoid(), false);

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(SigmoidBackward),
                output_id: output.id(),
                input_ids: vec![input.id()],
                saved_tensors: vec![output.clone()],
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

    /// @covers: Sigmoid::forward
    #[test]
    fn test_sigmoid_forward_maps_zero_to_half() {
        let mut sigmoid = Sigmoid::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let output = sigmoid.forward(&input).unwrap();
        let data = output.to_vec();
        assert!((data[0] - 0.5).abs() < 1e-6);
    }

    /// @covers: Sigmoid::parameters
    #[test]
    fn test_sigmoid_has_no_parameters() {
        let sig = Sigmoid::new();
        assert!(sig.parameters().is_empty());
    }

    /// @covers: Sigmoid::new
    #[test]
    fn test_sigmoid_default_matches_new() {
        let s1 = Sigmoid::new();
        let s2 = Sigmoid::default();
        assert_eq!(s1.parameters().len(), s2.parameters().len());
    }
}
