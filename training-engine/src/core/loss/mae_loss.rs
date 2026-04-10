use crate::api::error::SwetsResult;
use crate::api::loss::Loss;
use crate::api::tape::{self, BackwardOp, TapeEntry};
use crate::api::tensor::Tensor;

/// FR-602: Mean Absolute Error loss.
///
/// MAE = mean(|predictions - targets|)
pub struct MAELoss;

impl MAELoss {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MAELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for MAELoss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> SwetsResult<Tensor> {
        // MAE = mean(|pred - target|)
        let diff = predictions.sub_raw(targets)?;
        let abs_diff = Tensor::new(diff.inner().abs(), false);
        let mae_val = abs_diff.mean_all_raw();
        let output = Tensor::from_vec(vec![mae_val], vec![1])?;

        if tape::is_recording() {
            let entry = TapeEntry {
                backward_op: Box::new(MAEBackward {
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

/// Backward: d(MAE)/d(pred) = sign(pred - target) / n
struct MAEBackward {
    n: usize,
}

impl BackwardOp for MAEBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let predictions = &saved[0];
        let targets = &saved[1];

        let diff = predictions.sub_raw(targets).expect("mae backward sub");
        let diff_data = diff.to_vec();

        // sign: 1.0 if diff > 0, -1.0 if diff < 0, 0.0 if diff == 0
        let sign_data: Vec<f32> = diff_data
            .iter()
            .map(|&d| {
                if d > 0.0 {
                    1.0
                } else if d < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();

        let sign = Tensor::from_vec(sign_data, diff.shape().to_vec())
            .expect("mae backward sign tensor");

        // Scale by 1/n
        let scale = 1.0 / self.n as f32;
        let grad_pred = sign.mul_scalar_raw(scale);

        // Scale by upstream gradient
        let grad_val = grad_output.to_vec()[0];
        let grad_pred = grad_pred.mul_scalar_raw(grad_val);

        vec![grad_pred]
    }

    fn name(&self) -> &str {
        "MAEBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: MAELoss::new
    #[test]
    fn test_new_creates_instance() {
        let _loss = MAELoss::new();
    }

    /// @covers: MAELoss::default
    #[test]
    fn test_default_creates_instance() {
        let _loss = MAELoss::default();
    }

    /// @covers: MAELoss::forward
    #[test]
    fn test_mae_loss_identical_inputs_returns_zero() {
        let loss = MAELoss::new();
        let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let tgt = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let result = loss.forward(&pred, &tgt).unwrap();
        assert!(result.to_vec()[0].abs() < 1e-6);
    }

    /// @covers: MAELoss::forward
    #[test]
    fn test_mae_loss_known_value() {
        let loss = MAELoss::new();
        let pred = Tensor::from_vec(vec![1.0, 5.0], vec![2]).unwrap();
        let tgt = Tensor::from_vec(vec![3.0, 3.0], vec![2]).unwrap();
        let result = loss.forward(&pred, &tgt).unwrap();
        // MAE = mean(|1-3| + |5-3|) = mean(2 + 2) = 2.0
        assert!((result.to_vec()[0] - 2.0).abs() < 1e-6);
    }
}
