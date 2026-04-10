use std::f32::consts::PI;

use crate::api::optim::{LRScheduler, Optimizer};

/// FR-502: CosineAnnealingLR -- Cosine decay to eta_min
pub struct CosineAnnealingLR {
    initial_lr: f32,
    t_max: usize,
    eta_min: f32,
    current_step: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, t_max: usize, eta_min: f32) -> Self {
        Self {
            initial_lr,
            t_max,
            eta_min,
            current_step: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        optimizer.set_lr(self.get_lr());
    }

    fn get_lr(&self) -> f32 {
        let cos_value = (PI * self.current_step as f32 / self.t_max as f32).cos();
        self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + cos_value)
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

    /// @covers: CosineAnnealingLR::get_lr
    #[test]
    fn test_cosine_annealing_starts_at_initial_lr() {
        let scheduler = CosineAnnealingLR::new(0.1, 100, 0.0);
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    /// @covers: CosineAnnealingLR::step
    #[test]
    fn test_cosine_annealing_reaches_eta_min_at_t_max() {
        let mut scheduler = CosineAnnealingLR::new(0.1, 50, 0.001);
        let mut opt = MockOptimizer { lr: 0.1 };
        for _ in 0..50 { scheduler.step(&mut opt); }
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-6);
    }

    /// @covers: CosineAnnealingLR::step
    #[test]
    fn test_cosine_annealing_midpoint() {
        let mut scheduler = CosineAnnealingLR::new(0.1, 100, 0.0);
        let mut opt = MockOptimizer { lr: 0.1 };
        for _ in 0..50 { scheduler.step(&mut opt); }
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-4);
    }
}
