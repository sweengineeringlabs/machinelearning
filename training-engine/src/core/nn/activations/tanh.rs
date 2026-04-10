use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, TapeEntry};
use crate::api::tensor::Tensor;
use crate::core::ops::tanh::TanhBackward;

pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Tanh {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let output = Tensor::new(input.inner().tanh(), false);

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(TanhBackward),
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

    /// @covers: Tanh::forward
    #[test]
    fn test_tanh_forward_maps_zero_to_zero() {
        let mut tanh_layer = Tanh::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let output = tanh_layer.forward(&input).unwrap();
        let data = output.to_vec();
        assert!(data[0].abs() < 1e-6);
    }

    /// @covers: Tanh::parameters
    #[test]
    fn test_tanh_has_no_parameters() {
        let t = Tanh::new();
        assert!(t.parameters().is_empty());
    }

    /// @covers: Tanh::new
    #[test]
    fn test_tanh_default_matches_new() {
        let t1 = Tanh::new();
        let t2 = Tanh::default();
        assert_eq!(t1.parameters().len(), t2.parameters().len());
    }
}
