use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tensor::Tensor;

/// Sequential container that chains layers in order (FR-301).
///
/// Forward pass feeds each layer's output as the next layer's input.
/// Parameters are the union of all contained layers' parameters.
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    /// Returns the number of layers in the container.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true if the container has no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Layer for Sequential {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let mut x = input.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers
            .iter_mut()
            .flat_map(|l| l.parameters_mut())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::layers::linear::Linear;
    use crate::core::layers::activations::relu::ReLU;

    /// @covers: Sequential::new
    #[test]
    fn test_sequential_new_stores_layers() {
        let seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(ReLU::new()),
        ]);
        assert_eq!(seq.len(), 2);
        assert!(!seq.is_empty());
    }

    /// @covers: Sequential::forward
    #[test]
    fn test_sequential_forward_chains_layers() {
        let mut seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(ReLU::new()),
        ]);
        let input = Tensor::randn([2, 4]);
        let output = seq.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
    }

    /// @covers: Sequential::len
    #[test]
    fn test_len() {
        let seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
        ]);
        assert_eq!(seq.len(), 1);
    }

    /// @covers: Sequential::is_empty
    #[test]
    fn test_is_empty() {
        let seq = Sequential::new(vec![]);
        assert!(seq.is_empty());
    }

    /// @covers: Sequential::is_empty
    #[test]
    fn test_sequential_empty() {
        let seq = Sequential::new(vec![]);
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
    }

    /// @covers: Sequential::parameters
    #[test]
    fn test_sequential_parameters_aggregates_all_layers() {
        let seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(Linear::new(3, 2)),
        ]);
        // Two linear layers: each has weight + bias = 2 tensors, total = 4
        assert_eq!(seq.parameters().len(), 4);
    }

    /// @covers: Sequential::parameters_mut
    #[test]
    fn test_sequential_parameters_mut_aggregates_all_layers() {
        let mut seq = Sequential::new(vec![
            Box::new(Linear::new(4, 3)),
            Box::new(Linear::new(3, 2)),
        ]);
        assert_eq!(seq.parameters_mut().len(), 4);
    }
}
