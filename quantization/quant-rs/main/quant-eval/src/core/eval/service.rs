//! Content-correctness metrics for comparing a reference tensor against a
//! candidate (e.g. dequantized) tensor.
//!
//! The metrics here answer the question "does the quantized weight still
//! encode the same values?" and are the primary gate for declaring a
//! backend correct. Latency is tracked separately.

use candle_core::Tensor;
use quant_api::QuantError;

/// Quantitative comparison between a reference tensor and a candidate
/// tensor of identical shape.
///
/// All fields are in their natural units:
/// * `snr_db` is in decibels (`10 * log10(signal / noise)`).
/// * `cosine` is unitless in `[-1, 1]`.
/// * `mse` and `max_abs_error` are in the same unit as the input tensors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    /// Signal-to-noise ratio in decibels. `+inf` when candidate equals
    /// reference exactly. `-inf` when reference is all-zero but
    /// candidate is not.
    pub snr_db: f32,
    /// Cosine similarity of the flattened tensors. Defined as `0.0` when
    /// either vector has zero norm (a degenerate case we surface rather
    /// than panic on).
    pub cosine: f32,
    /// Mean squared error: `(1/n) * sum((a_i - b_i)^2)`.
    pub mse: f32,
    /// Maximum elementwise absolute difference.
    pub max_abs_error: f32,
}

/// Computes content-correctness metrics between two tensors.
///
/// Held as a unit struct rather than a free function so callers can plug
/// alternative implementations behind a trait later (per SEA/OOP
/// conventions) without changing call sites.
#[derive(Debug, Default, Clone, Copy)]
pub struct EvalService;

impl EvalService {
    /// Creates a new evaluator. Stateless today; kept as a constructor so
    /// future configuration (tolerance, reduction axis, ...) can be
    /// added without breaking callers.
    pub fn new() -> Self {
        Self
    }

    /// Compares `reference` against `candidate` and returns the metric
    /// bundle.
    ///
    /// # Errors
    /// Returns [`QuantError::Invalid`] if the shapes differ, if either
    /// tensor is empty, or if the underlying backend rejects the
    /// `flatten_all` / `to_vec1::<f32>` conversion (wrapped via
    /// [`QuantError::Tensor`] through `From<candle_core::Error>`).
    pub fn calculate_metrics(
        &self,
        reference: &Tensor,
        candidate: &Tensor,
    ) -> Result<Metrics, QuantError> {
        let ref_shape = reference.shape().dims().to_vec();
        let cand_shape = candidate.shape().dims().to_vec();
        if ref_shape != cand_shape {
            return Err(QuantError::Invalid(format!(
                "shape mismatch: reference {:?} vs candidate {:?}",
                ref_shape, cand_shape
            )));
        }

        let a: Vec<f32> = reference.flatten_all()?.to_vec1::<f32>()?;
        let b: Vec<f32> = candidate.flatten_all()?.to_vec1::<f32>()?;

        if a.is_empty() {
            return Err(QuantError::Invalid(
                "cannot compute metrics over empty tensor".to_string(),
            ));
        }
        // Shapes match and a is non-empty, but double-check in case an
        // exotic shape flattens to zero elements on only one side.
        if a.len() != b.len() {
            return Err(QuantError::Invalid(format!(
                "flattened length mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        let n = a.len() as f32;

        let mut signal_power = 0.0f64; // f64 accumulators — avoid float drift on large tensors
        let mut noise_power = 0.0f64;
        let mut dot = 0.0f64;
        let mut norm_a_sq = 0.0f64;
        let mut norm_b_sq = 0.0f64;
        let mut max_abs_err = 0.0f32;

        for (&ai, &bi) in a.iter().zip(b.iter()) {
            let diff = ai - bi;
            let diff_sq = (diff as f64) * (diff as f64);
            signal_power += (ai as f64) * (ai as f64);
            noise_power += diff_sq;
            dot += (ai as f64) * (bi as f64);
            norm_a_sq += (ai as f64) * (ai as f64);
            norm_b_sq += (bi as f64) * (bi as f64);
            let ae = diff.abs();
            if ae > max_abs_err {
                max_abs_err = ae;
            }
        }

        let mse = (noise_power / n as f64) as f32;

        let snr_db = if noise_power == 0.0 {
            f32::INFINITY
        } else if signal_power == 0.0 {
            f32::NEG_INFINITY
        } else {
            (10.0 * (signal_power / noise_power).log10()) as f32
        };

        let norm_a = norm_a_sq.sqrt();
        let norm_b = norm_b_sq.sqrt();
        let cosine = if norm_a == 0.0 || norm_b == 0.0 {
            // Degenerate: one vector is the zero vector. Cosine is
            // mathematically undefined; report 0.0 so callers can still
            // aggregate without special-casing NaN.
            0.0
        } else {
            (dot / (norm_a * norm_b)) as f32
        };

        Ok(Metrics {
            snr_db,
            cosine,
            mse,
            max_abs_error: max_abs_err,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn tensor_1d(values: &[f32]) -> Tensor {
        Tensor::from_slice(values, (values.len(),), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_metrics_for_identical_tensors_have_zero_error_and_unit_cosine() {
        let a = tensor_1d(&[1.0, 2.0, 3.0, 4.0]);
        let b = tensor_1d(&[1.0, 2.0, 3.0, 4.0]);

        let m = EvalService::new().calculate_metrics(&a, &b).unwrap();

        assert_eq!(m.mse, 0.0, "MSE must be exactly zero for identical tensors");
        assert_eq!(
            m.max_abs_error, 0.0,
            "max_abs_error must be exactly zero for identical tensors"
        );
        assert!(
            (m.cosine - 1.0).abs() < 1e-6,
            "cosine should be ~1.0 for identical tensors, got {}",
            m.cosine
        );
        assert!(
            m.snr_db.is_infinite() && m.snr_db.is_sign_positive(),
            "snr_db should be +inf for identical tensors, got {}",
            m.snr_db
        );
    }

    #[test]
    fn test_metrics_for_orthogonal_tensors_have_zero_cosine() {
        let a = tensor_1d(&[1.0, 0.0]);
        let b = tensor_1d(&[0.0, 1.0]);

        let m = EvalService::new().calculate_metrics(&a, &b).unwrap();

        assert!(
            m.cosine.abs() < 1e-6,
            "cosine of orthogonal vectors should be ~0.0, got {}",
            m.cosine
        );
    }

    #[test]
    fn test_metrics_for_anti_parallel_tensors_have_negative_unit_cosine() {
        let a = tensor_1d(&[1.0, 2.0, 3.0]);
        let b = tensor_1d(&[-1.0, -2.0, -3.0]);

        let m = EvalService::new().calculate_metrics(&a, &b).unwrap();

        assert!(
            (m.cosine + 1.0).abs() < 1e-6,
            "cosine of anti-parallel vectors should be ~-1.0, got {}",
            m.cosine
        );
    }

    #[test]
    fn test_metrics_with_known_quantization_noise_yields_expected_snr() {
        let a = tensor_1d(&[1.0, 2.0, 3.0, 4.0]);
        let b = tensor_1d(&[1.1, 2.1, 3.1, 4.1]);

        let m = EvalService::new().calculate_metrics(&a, &b).unwrap();

        // signal = 1 + 4 + 9 + 16 = 30
        // noise  = 4 * 0.01       = 0.04
        // snr_db = 10 * log10(750) ≈ 28.7506
        let expected_snr = 10.0f32 * 750.0f32.log10();
        assert!(
            (m.snr_db - expected_snr).abs() < 0.01,
            "snr_db should be ~{:.4} dB, got {}",
            expected_snr,
            m.snr_db
        );

        // mse = 0.04 / 4 = 0.01
        assert!(
            (m.mse - 0.01).abs() < 1e-6,
            "mse should be ~0.01, got {}",
            m.mse
        );

        // All four elementwise diffs are 0.1 in magnitude.
        assert!(
            (m.max_abs_error - 0.1).abs() < 1e-6,
            "max_abs_error should be ~0.1, got {}",
            m.max_abs_error
        );
    }

    #[test]
    fn test_calculate_metrics_rejects_shape_mismatch() {
        let a = tensor_1d(&[1.0, 2.0, 3.0, 4.0]);
        let b = tensor_1d(&[1.0, 2.0, 3.0]);

        let err = EvalService::new()
            .calculate_metrics(&a, &b)
            .expect_err("expected shape-mismatch error");

        match err {
            QuantError::Invalid(msg) => {
                assert!(
                    msg.contains("shape"),
                    "error message should mention shape, got {}",
                    msg
                );
            }
            other => panic!("expected QuantError::Invalid, got {:?}", other),
        }
    }

    #[test]
    fn test_calculate_metrics_rejects_empty_tensor() {
        let empty_a: Vec<f32> = vec![];
        let empty_b: Vec<f32> = vec![];
        let a = Tensor::from_slice(&empty_a, (0,), &Device::Cpu).unwrap();
        let b = Tensor::from_slice(&empty_b, (0,), &Device::Cpu).unwrap();

        let err = EvalService::new()
            .calculate_metrics(&a, &b)
            .expect_err("expected empty-tensor error");

        match err {
            QuantError::Invalid(msg) => {
                assert!(
                    msg.to_lowercase().contains("empty"),
                    "error message should mention empty, got {}",
                    msg
                );
            }
            other => panic!("expected QuantError::Invalid, got {:?}", other),
        }
    }
}
