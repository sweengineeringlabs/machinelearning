use crate::api::optimizer::Optimizer;

pub trait LRScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer);
    fn get_lr(&self) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::error::SwetsResult;
    use crate::api::tensor::Tensor;

    struct MockOptimizer {
        lr: f32,
    }

    impl Optimizer for MockOptimizer {
        fn step(&mut self, _params: &mut [&mut Tensor]) -> SwetsResult<()> {
            Ok(())
        }
        fn lr(&self) -> f32 {
            self.lr
        }
        fn set_lr(&mut self, lr: f32) {
            self.lr = lr;
        }
    }

    struct FixedScheduler {
        lr: f32,
    }

    impl LRScheduler for FixedScheduler {
        fn step(&mut self, optimizer: &mut dyn Optimizer) {
            optimizer.set_lr(self.lr);
        }
        fn get_lr(&self) -> f32 {
            self.lr
        }
    }

    #[test]
    fn test_lr_scheduler_step_updates_optimizer_lr() {
        let mut scheduler = FixedScheduler { lr: 0.05 };
        let mut opt = MockOptimizer { lr: 0.1 };
        scheduler.step(&mut opt);
        assert!((opt.lr() - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lr_scheduler_get_lr_returns_scheduled_value() {
        let scheduler = FixedScheduler { lr: 0.05 };
        assert!((scheduler.get_lr() - 0.05).abs() < f32::EPSILON);
    }
}
