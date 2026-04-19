use std::collections::HashMap;
use swe_ml_tensor::Tensor;

use crate::api::error::{WeightError, WeightResult};
use crate::api::traits::WeightGuard;

/// Default weight-presence validator. Returns `WeightError::Missing`
/// on the first expected name that isn't in the map.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultWeightGuard;

impl WeightGuard for DefaultWeightGuard {
    fn validate(
        &self,
        weights: &HashMap<String, Tensor>,
        expected: &[&str],
    ) -> WeightResult<()> {
        for name in expected {
            if !weights.contains_key(*name) {
                return Err(WeightError::Missing((*name).to_string()));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_ok_when_all_present() {
        let mut weights = HashMap::new();
        weights.insert("a".to_string(), Tensor::zeros(vec![1]));
        weights.insert("b".to_string(), Tensor::zeros(vec![1]));

        let guard = DefaultWeightGuard;
        assert!(guard.validate(&weights, &["a", "b"]).is_ok());
    }

    #[test]
    fn test_validate_rejects_missing_weight() {
        let mut weights = HashMap::new();
        weights.insert("a".to_string(), Tensor::zeros(vec![1]));

        let guard = DefaultWeightGuard;
        let result = guard.validate(&weights, &["a", "b"]);
        assert!(matches!(result, Err(WeightError::Missing(name)) if name == "b"));
    }

    #[test]
    fn test_validate_returns_first_missing() {
        let weights: HashMap<String, Tensor> = HashMap::new();

        let guard = DefaultWeightGuard;
        let result = guard.validate(&weights, &["first", "second"]);
        assert!(matches!(result, Err(WeightError::Missing(name)) if name == "first"));
    }
}
