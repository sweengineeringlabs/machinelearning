use std::f32::consts::PI;

use crate::api::optim::{LRScheduler, Optimizer};

/// FR-503: WarmupCosineScheduler -- Linear warmup then cosine decay
pub struct WarmupCosineScheduler {
    initial_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    eta_min: f32,
    current_step: usize,
}

impl WarmupCosineScheduler {
    pub fn new(
        initial_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
        eta_min: f32,
    ) -> Self {
        Self {
            initial_lr,
            warmup_steps,
            total_steps,
            eta_min,
            current_step: 0,
        }
    }
}

impl LRScheduler for WarmupCosineScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        optimizer.set_lr(self.get_lr());
    }

    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            self.initial_lr * self.current_step as f32 / self.warmup_steps as f32
        } else {
            let decay_steps = self.total_steps - self.warmup_steps;
            let progress = (self.current_step - self.warmup_steps) as f32 / decay_steps as f32;
            let cos_value = (PI * progress).cos();
            self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + cos_value)
        }
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

    /// @covers: WarmupCosineScheduler::get_lr
    #[test]
    fn test_warmup_cosine_starts_at_zero() {
        let scheduler = WarmupCosineScheduler::new(0.1, 10, 100, 0.0);
        assert!((scheduler.get_lr()).abs() < 1e-6);
    }

    /// @covers: WarmupCosineScheduler::step
    #[test]
    fn test_warmup_cosine_linear_phase() {
        let mut scheduler = WarmupCosineScheduler::new(0.1, 10, 100, 0.0);
        let mut opt = MockOptimizer { lr: 0.0 };
        for _ in 0..5 { scheduler.step(&mut opt); }
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-6);
    }

    /// @covers: WarmupCosineScheduler::step
    #[test]
    fn test_warmup_cosine_reaches_peak_at_warmup_end() {
        let mut scheduler = WarmupCosineScheduler::new(0.1, 10, 100, 0.0);
        let mut opt = MockOptimizer { lr: 0.0 };
        for _ in 0..10 { scheduler.step(&mut opt); }
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    /// @covers: WarmupCosineScheduler::step
    #[test]
    fn test_warmup_cosine_decays_to_eta_min() {
        let mut scheduler = WarmupCosineScheduler::new(0.1, 10, 100, 0.001);
        let mut opt = MockOptimizer { lr: 0.0 };
        for _ in 0..100 { scheduler.step(&mut opt); }
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-6);
    }
}
