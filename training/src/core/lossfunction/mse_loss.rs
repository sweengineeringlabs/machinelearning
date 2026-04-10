use crate::api::error::SwetsResult;
use crate::api::loss::Loss;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;

pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for MSELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> SwetsResult<Tensor> {
        // MSE = mean((pred - target)^2)
        let diff = predictions.sub_raw(targets)?;
        let sq = diff.pow_raw(2.0);
        let mse_val = sq.mean_all_raw();
        let output = Tensor::from_vec(vec![mse_val], vec![1])?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(MSEBackward {
                    n: predictions.numel(),
                }),
                output_id: output.id(),
                input_ids: vec![predictions.id()],
                saved_tensors: vec![predictions.clone(), targets.clone()],
            };
            tape::record_op(entry);
        }

        Ok(output)
    }
}

/// Backward: d(MSE)/d(pred) = 2 * (pred - target) / n
struct MSEBackward {
    n: usize,
}

impl BackwardOp for MSEBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let predictions = &saved[0];
        let targets = &saved[1];

        let diff = predictions.sub_raw(targets).expect("mse backward sub");
        let scale = 2.0 / self.n as f32;
        let grad_pred = diff.mul_scalar_raw(scale);

        // Scale by upstream gradient
        let grad_val = grad_output.to_vec()[0];
        let grad_pred = grad_pred.mul_scalar_raw(grad_val);

        vec![grad_pred]
    }

    fn name(&self) -> &str {
        "MSEBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: MSELoss::new
    #[test]
    fn test_new_creates_instance() {
        let _loss = MSELoss::new();
    }

    /// @covers: MSELoss::default
    #[test]
    fn test_default_creates_instance() {
        let _loss = MSELoss::default();
    }

    /// @covers: MSELoss::forward
    #[test]
    fn test_mse_loss_identical_inputs_returns_zero() {
        let loss = MSELoss::new();
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let tgt = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = loss.forward(&pred, &tgt).unwrap();
        assert!(result.to_vec()[0].abs() < 1e-6);
    }

    /// @covers: MSELoss::forward
    #[test]
    fn test_mse_loss_known_value() {
        let loss = MSELoss::new();
        let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let tgt = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
        let result = loss.forward(&pred, &tgt).unwrap();
        // MSE = mean((1-3)^2 + (2-4)^2) = mean(4 + 4) = 4.0
        assert!((result.to_vec()[0] - 4.0).abs() < 1e-6);
    }
}
