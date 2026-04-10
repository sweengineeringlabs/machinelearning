use crate::api::optim::{LRScheduler, Optimizer};

/// FR-501: StepLR -- Decay by gamma every step_size steps
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
    current_step: usize,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        optimizer.set_lr(self.get_lr());
    }

    fn get_lr(&self) -> f32 {
        let exponent = (self.current_step / self.step_size) as u32;
        self.initial_lr * self.gamma.powi(exponent as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::error::SwetsResult;
    use crate::api::tensor::Tensor;

    struct MockOptimizer { lr: f32 }
    impl Optimizer for MockOptimizer {
        fn step(&mut self, _params: &mut [&mut Tensor]) -> SwetsResult<()> { Ok(()) }
        fn lr(&self) -> f32 { self.lr }
        fn set_lr(&mut self, lr: f32) { self.lr = lr; }
    }

    /// @covers: StepLR::get_lr
    #[test]
    fn test_step_lr_initial_lr_unchanged_before_step_size() {
        let scheduler = StepLR::new(0.1, 10, 0.5);
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    /// @covers: StepLR::step
    #[test]
    fn test_step_lr_decays_at_step_size_boundary() {
        let mut scheduler = StepLR::new(0.1, 5, 0.5);
        let mut opt = MockOptimizer { lr: 0.1 };
        for _ in 0..5 { scheduler.step(&mut opt); }
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-6);
        assert!((opt.lr() - 0.05).abs() < 1e-6);
    }

    /// @covers: StepLR::step
    #[test]
    fn test_step_lr_multiple_decays() {
        let mut scheduler = StepLR::new(0.1, 3, 0.5);
        let mut opt = MockOptimizer { lr: 0.1 };
        for _ in 0..9 { scheduler.step(&mut opt); }
        assert!((scheduler.get_lr() - 0.0125).abs() < 1e-6);
    }

    /// @covers: StepLR::new
    #[test]
    fn test_step_lr_new_creates_instance() {
        let scheduler = StepLR::new(0.05, 10, 0.1);
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-6);
    }
}
