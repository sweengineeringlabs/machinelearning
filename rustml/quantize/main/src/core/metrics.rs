use crate::api::types::TensorReport;

/// Compute quantization quality metrics by comparing original and reconstructed values.
pub(crate) fn compute_metrics(
    name: &str,
    original: &[f32],
    reconstructed: &[f32],
    original_dtype: &str,
    target_dtype: &str,
    original_bytes: u64,
    quantized_bytes: u64,
) -> TensorReport {
    let n = original.len();
    if n == 0 || n != reconstructed.len() {
        return TensorReport {
            name: name.to_string(),
            original_dtype: original_dtype.to_string(),
            target_dtype: target_dtype.to_string(),
            original_bytes,
            quantized_bytes,
            mse: None,
            max_abs_error: None,
            snr_db: None,
        };
    }

    let mut sum_sq_error = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut sum_sq_signal = 0.0f64;

    for i in 0..n {
        let orig = original[i] as f64;
        let recon = reconstructed[i] as f64;
        let err = orig - recon;
        sum_sq_error += err * err;
        let abs_err = err.abs();
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        sum_sq_signal += orig * orig;
    }

    let mse = sum_sq_error / n as f64;
    let snr_db = if sum_sq_error > 0.0 {
        10.0 * (sum_sq_signal / sum_sq_error).log10()
    } else {
        f64::INFINITY
    };

    TensorReport {
        name: name.to_string(),
        original_dtype: original_dtype.to_string(),
        target_dtype: target_dtype.to_string(),
        original_bytes,
        quantized_bytes,
        mse: Some(mse),
        max_abs_error: Some(max_abs),
        snr_db: Some(snr_db),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_metrics_identical_produces_zero_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let report = compute_metrics("test", &data, &data, "F32", "F32", 16, 16);
        assert_eq!(report.mse.unwrap(), 0.0);
        assert_eq!(report.max_abs_error.unwrap(), 0.0);
    }

    #[test]
    fn test_compute_metrics_with_error() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let recon = vec![1.1, 2.0, 2.9, 4.0];
        let report = compute_metrics("test", &orig, &recon, "F32", "Q8_0", 16, 8);
        assert!(report.mse.unwrap() > 0.0);
        assert!(report.max_abs_error.unwrap() > 0.09);
        assert!(report.snr_db.unwrap() > 0.0);
    }
}
