use crate::api::error::SwetsResult;
use crate::api::tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self, params: &mut [&mut Tensor]) -> SwetsResult<()>;
    fn lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockOptimizer {
        learning_rate: f32,
    }

    impl Optimizer for MockOptimizer {
        fn step(&mut self, _params: &mut [&mut Tensor]) -> SwetsResult<()> {
            Ok(())
        }
        fn lr(&self) -> f32 {
            self.learning_rate
        }
        fn set_lr(&mut self, lr: f32) {
            self.learning_rate = lr;
        }
    }

    #[test]
    fn test_optimizer_lr_returns_configured_value() {
        let opt = MockOptimizer { learning_rate: 0.01 };
        assert!((opt.lr() - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_optimizer_set_lr_updates_value() {
        let mut opt = MockOptimizer { learning_rate: 0.01 };
        opt.set_lr(0.001);
        assert!((opt.lr() - 0.001).abs() < f32::EPSILON);
    }
}
