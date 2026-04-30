use crate::api::error::{EmbeddingError, EmbeddingResult};
use crate::api::traits::Normalize;

/// L2 (Euclidean) unit-length normalization.
///
/// After `normalize`, cosine similarity between two vectors equals their dot product.
/// A zero vector is returned unchanged (division is skipped).
/// Non-finite input (NaN or ±inf) is rejected.
#[derive(Debug, Default, Clone, Copy)]
pub struct L2Normalize;

impl Normalize for L2Normalize {
    fn normalize(&self, v: &mut [f32]) -> EmbeddingResult<()> {
        for (i, x) in v.iter().enumerate() {
            if !x.is_finite() {
                return Err(EmbeddingError::NonFinite(format!(
                    "element {} is {}",
                    i, x
                )));
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_3_4_returns_0_6_and_0_8() {
        let mut v = vec![3.0, 4.0];
        L2Normalize.normalize(&mut v).unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_arbitrary_vector_produces_unit_norm() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        L2Normalize.normalize(&mut v).unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector_unchanged() {
        let mut v = vec![0.0, 0.0, 0.0];
        L2Normalize.normalize(&mut v).unwrap();
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_normalize_nan_input_rejected() {
        let mut v = vec![1.0, f32::NAN, 2.0];
        let result = L2Normalize.normalize(&mut v);
        assert!(matches!(result, Err(EmbeddingError::NonFinite(_))));
    }

    #[test]
    fn test_normalize_positive_infinity_rejected() {
        let mut v = vec![1.0, f32::INFINITY];
        let result = L2Normalize.normalize(&mut v);
        assert!(matches!(result, Err(EmbeddingError::NonFinite(_))));
    }

    #[test]
    fn test_normalize_negative_infinity_rejected() {
        let mut v = vec![f32::NEG_INFINITY, 1.0];
        let result = L2Normalize.normalize(&mut v);
        assert!(matches!(result, Err(EmbeddingError::NonFinite(_))));
    }

    #[test]
    fn test_normalize_leaves_input_untouched_when_rejected() {
        let mut v = vec![1.0, f32::NAN, 2.0];
        let _ = L2Normalize.normalize(&mut v);
        assert_eq!(v[0], 1.0);
        assert!(v[1].is_nan());
        assert_eq!(v[2], 2.0);
    }
}
