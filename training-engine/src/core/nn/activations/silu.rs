use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tape::{self, TapeEntry};
use crate::api::tensor::Tensor;
use crate::core::nn::activations::silu_backward::SiLUBackward;

pub struct SiLU;

impl SiLU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for SiLU {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let x_data = input.to_vec();

        let output_data: Vec<f32> = x_data
            .iter()
            .map(|&x| {
                let sig = 1.0 / (1.0 + (-x).exp());
                x * sig
            })
            .collect();

        let output = Tensor::from_vec(output_data, input.shape().to_vec())?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(SiLUBackward),
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

    /// @covers: SiLU::forward
    #[test]
    fn test_silu_forward_maps_zero_to_zero() {
        let mut silu = SiLU::new();
        let input = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let output = silu.forward(&input).unwrap();
        let data = output.to_vec();
        assert!(data[0].abs() < 1e-6);
    }

    /// @covers: SiLU::parameters
    #[test]
    fn test_silu_has_no_parameters() {
        let silu = SiLU::new();
        assert!(silu.parameters().is_empty());
    }

    /// @covers: SiLU::new
    #[test]
    fn test_silu_default_matches_new() {
        let s1 = SiLU::new();
        let s2 = SiLU::default();
        assert_eq!(s1.parameters().len(), s2.parameters().len());
    }
}
