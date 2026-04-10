use crate::api::error::SwetsResult;
use crate::api::tensor::Tensor;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    fn parameter_count(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyLayer {
        weight: Tensor,
    }

    impl DummyLayer {
        fn new() -> Self {
            Self {
                weight: Tensor::ones(vec![3, 2]),
            }
        }
    }

    impl Layer for DummyLayer {
        fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
            Ok(input.clone())
        }
        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.weight]
        }
        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.weight]
        }
    }

    #[test]
    fn test_parameter_count_sums_numel_of_all_params() {
        let layer = DummyLayer::new();
        assert_eq!(layer.parameter_count(), 6); // 3 * 2
    }

    #[test]
    fn test_layer_forward_returns_ok() {
        let mut layer = DummyLayer::new();
        let input = Tensor::ones(vec![2, 3]);
        let result = layer.forward(&input);
        assert!(result.is_ok());
    }
}
