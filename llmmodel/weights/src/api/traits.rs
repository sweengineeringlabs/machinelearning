use std::collections::HashMap;
use swe_ml_tensor::Tensor;

use crate::api::error::WeightResult;

/// Map HuggingFace tensor names to architecture-internal names.
pub trait WeightMapper {
    fn map_weights(
        &self,
        weights: HashMap<String, Tensor>,
    ) -> WeightResult<HashMap<String, Tensor>>;

    /// Top-level weight names the architecture requires.
    fn expected_weights(&self) -> Vec<&'static str>;
}

/// Check that a weight map satisfies an expected-name list.
pub trait WeightGuard {
    fn validate(
        &self,
        weights: &HashMap<String, Tensor>,
        expected: &[&str],
    ) -> WeightResult<()>;
}
