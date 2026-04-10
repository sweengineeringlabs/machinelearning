use crate::api::error::SwetsResult;
use crate::api::tensor::Tensor;

pub trait Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> SwetsResult<Tensor>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroLoss;

    impl Loss for ZeroLoss {
        fn forward(&self, _predictions: &Tensor, _targets: &Tensor) -> SwetsResult<Tensor> {
            Ok(Tensor::zeros(vec![1]))
        }
    }

    #[test]
    fn test_loss_trait_forward_returns_scalar() {
        let loss = ZeroLoss;
        let pred = Tensor::ones(vec![4]);
        let tgt = Tensor::zeros(vec![4]);
        let result = loss.forward(&pred, &tgt).unwrap();
        assert_eq!(result.shape(), &[1]);
    }
}
