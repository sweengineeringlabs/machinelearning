use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, TapeEntry};
use crate::api::tensor::Tensor;
use crate::core::layers::activations::gelu_backward::GELUBackward;
use swe_ml_activation::{Activation, Gelu as GeluImpl};

pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GELU {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let core_output = GeluImpl.forward(input.inner())
            .map_err(|e| crate::api::error::SwetsError::Layer(e.to_string()))?;

        let output = Tensor::from_vec(core_output.to_vec(), input.shape().to_vec())?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(GELUBackward),
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

    /// @covers: GELU::forward
    #[test]
    fn test_gelu_forward_maps_zero_to_zero() {
        let mut gelu = GELU::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let output = gelu.forward(&input).unwrap();
        let data = output.to_vec();
        assert!(data[0].abs() < 1e-6);
    }

    /// @covers: GELU::forward
    #[test]
    fn test_gelu_forward_positive_input_returns_positive() {
        let mut gelu = GELU::new();
        let input = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let output = gelu.forward(&input).unwrap();
        let data = output.to_vec();
        assert!(data[0] > 0.0);
    }

    /// @covers: GELU::parameters
    #[test]
    fn test_gelu_has_no_parameters() {
        let gelu = GELU::new();
        assert!(gelu.parameters().is_empty());
    }

    /// @covers: GELU::new
    #[test]
    fn test_gelu_default_matches_new() {
        let g1 = GELU::new();
        let g2 = GELU::default();
        assert_eq!(g1.parameters().len(), g2.parameters().len());
    }
}
