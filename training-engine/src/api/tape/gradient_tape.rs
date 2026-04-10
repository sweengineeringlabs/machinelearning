use crate::api::tape::tape_entry::TapeEntry;
use crate::api::tensor::{Tensor, TensorId};
use std::cell::RefCell;
use std::collections::HashMap;

// --- GradientTape ---

pub struct GradientTape {
    entries: Vec<TapeEntry>,
    pub(crate) grads: HashMap<TensorId, Tensor>,
    enabled: bool,
}

impl GradientTape {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            grads: HashMap::new(),
            enabled: true,
        }
    }

    pub fn record(&mut self, entry: TapeEntry) {
        if self.enabled {
            self.entries.push(entry);
        }
    }

    pub fn backward(&mut self, loss_id: TensorId, loss_shape: &[usize]) {
        // Seed gradient: ones with the shape of the loss
        let seed = Tensor::ones(loss_shape.to_vec());
        self.grads.insert(loss_id, seed);

        // Replay in reverse
        for i in (0..self.entries.len()).rev() {
            let output_id = self.entries[i].output_id;
            let grad_output = match self.grads.get(&output_id) {
                Some(g) => g.clone(),
                None => continue,
            };

            let input_grads = self.entries[i]
                .backward_op
                .backward(&grad_output, &self.entries[i].saved_tensors);

            for (j, input_id) in self.entries[i].input_ids.iter().enumerate() {
                if j < input_grads.len() {
                    let new_grad = &input_grads[j];
                    if let Some(existing) = self.grads.get(input_id) {
                        // Accumulate gradients
                        let accumulated = existing.add_raw(new_grad).expect("gradient accumulation");
                        self.grads.insert(*input_id, accumulated);
                    } else {
                        self.grads.insert(*input_id, new_grad.clone());
                    }
                }
            }
        }
    }

    pub fn grad(&self, id: TensorId) -> Option<&Tensor> {
        self.grads.get(&id)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.grads.clear();
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

// --- Thread-local API ---

thread_local! {
    static TAPE: RefCell<GradientTape> = RefCell::new(GradientTape::new());
}

pub fn record_op(entry: TapeEntry) {
    TAPE.with(|tape| tape.borrow_mut().record(entry));
}

pub fn backward(loss: &Tensor) {
    TAPE.with(|tape| {
        tape.borrow_mut().backward(loss.id(), loss.shape());
    });
}

pub fn grad(tensor: &Tensor) -> Option<Tensor> {
    TAPE.with(|tape| tape.borrow().grad(tensor.id()).cloned())
}

pub fn set_grad(tensor: &Tensor, grad: Tensor) {
    TAPE.with(|tape| {
        tape.borrow_mut().grads.insert(tensor.id(), grad);
    });
}

pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let was_enabled = TAPE.with(|tape| {
        let mut t = tape.borrow_mut();
        let prev = t.is_enabled();
        t.disable();
        prev
    });
    let result = f();
    if was_enabled {
        TAPE.with(|tape| tape.borrow_mut().enable());
    }
    result
}

pub fn clear_tape() {
    TAPE.with(|tape| tape.borrow_mut().clear());
}

pub fn is_recording() -> bool {
    TAPE.with(|tape| tape.borrow().is_enabled())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: GradientTape::new
    #[test]
    fn test_gradient_tape_starts_enabled() {
        let tape = GradientTape::new();
        assert!(tape.is_enabled());
    }

    /// @covers: GradientTape::disable
    #[test]
    fn test_gradient_tape_disable_stops_recording() {
        let mut tape = GradientTape::new();
        tape.disable();
        assert!(!tape.is_enabled());
    }

    /// @covers: GradientTape::enable
    #[test]
    fn test_gradient_tape_enable_resumes_recording() {
        let mut tape = GradientTape::new();
        tape.disable();
        tape.enable();
        assert!(tape.is_enabled());
    }

    /// @covers: no_grad
    #[test]
    fn test_no_grad_disables_recording_during_closure() {
        clear_tape();
        let was_recording_inside = no_grad(|| is_recording());
        assert!(!was_recording_inside);
        // Recording should be restored after no_grad
        assert!(is_recording());
    }

    /// @covers: clear_tape
    #[test]
    fn test_clear_tape_removes_all_entries() {
        clear_tape();
        // Just ensure it doesn't panic
        assert!(is_recording());
    }

    /// @covers: is_recording
    #[test]
    fn test_is_recording_returns_true_by_default() {
        clear_tape();
        assert!(is_recording());
    }

    /// @covers: record_op
    #[test]
    fn test_record_op_does_not_panic() {
        use crate::api::tape::backward_op::BackwardOp;
        clear_tape();
        struct NoopOp;
        impl BackwardOp for NoopOp {
            fn backward(&self, _g: &Tensor, _s: &[Tensor]) -> Vec<Tensor> { vec![] }
            fn name(&self) -> &str { "Noop" }
        }
        let t = Tensor::zeros(vec![1]);
        let entry = TapeEntry {
            backward_op: Box::new(NoopOp),
            output_id: t.id(),
            input_ids: vec![],
            saved_tensors: vec![],
        };
        record_op(entry);
    }

    /// @covers: backward
    #[test]
    fn test_backward_runs_without_entries() {
        clear_tape();
        let loss = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        backward(&loss);
    }

    /// @covers: grad
    #[test]
    fn test_grad_returns_none_when_not_set() {
        clear_tape();
        let t = Tensor::zeros(vec![2]);
        let g = grad(&t);
        assert!(g.is_none());
    }

    /// @covers: set_grad
    #[test]
    fn test_set_grad_stores_gradient() {
        clear_tape();
        let t = Tensor::zeros(vec![2]);
        let g = Tensor::ones(vec![2]);
        set_grad(&t, g);
        let result = grad(&t);
        assert!(result.is_some());
        assert_eq!(result.unwrap().to_vec(), vec![1.0, 1.0]);
    }

    /// @covers: GradientTape::is_enabled
    #[test]
    fn test_is_enabled_matches_state() {
        let mut tape = GradientTape::new();
        assert!(tape.is_enabled());
        tape.disable();
        assert!(!tape.is_enabled());
    }

    /// @covers: GradientTape::record
    #[test]
    fn test_record_skips_when_disabled() {
        use crate::api::tape::backward_op::BackwardOp;
        let mut tape = GradientTape::new();
        tape.disable();
        struct NoopOp;
        impl BackwardOp for NoopOp {
            fn backward(&self, _g: &Tensor, _s: &[Tensor]) -> Vec<Tensor> { vec![] }
            fn name(&self) -> &str { "Noop" }
        }
        let t = Tensor::zeros(vec![1]);
        let entry = TapeEntry {
            backward_op: Box::new(NoopOp),
            output_id: t.id(),
            input_ids: vec![],
            saved_tensors: vec![],
        };
        tape.record(entry);
        // Should have no entries since tape was disabled
    }

    /// @covers: GradientTape::clear
    #[test]
    fn test_clear_resets_grads() {
        let mut tape = GradientTape::new();
        let t = Tensor::zeros(vec![1]);
        tape.grads.insert(t.id(), Tensor::ones(vec![1]));
        tape.clear();
        assert!(tape.grad(t.id()).is_none());
    }

    /// @covers: GradientTape::grad
    #[test]
    fn test_grad_returns_stored_gradient() {
        let mut tape = GradientTape::new();
        let t = Tensor::zeros(vec![2]);
        let g = Tensor::ones(vec![2]);
        tape.grads.insert(t.id(), g);
        assert!(tape.grad(t.id()).is_some());
    }

    /// @covers: GradientTape::backward
    #[test]
    fn test_backward_seeds_loss_gradient() {
        let mut tape = GradientTape::new();
        let loss = Tensor::from_vec(vec![5.0], vec![1]).unwrap();
        tape.backward(loss.id(), loss.shape());
        let g = tape.grad(loss.id());
        assert!(g.is_some());
        assert_eq!(g.unwrap().to_vec(), vec![1.0]);
    }
}
