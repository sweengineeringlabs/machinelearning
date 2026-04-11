/// Configuration for normalization layers.
#[derive(Debug, Clone)]
pub struct NormConfig {
    /// Size of the normalized dimension.
    pub normalized_shape: usize,
    /// Small constant for numerical stability.
    pub eps: f32,
}

impl NormConfig {
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-5,
        }
    }

    pub fn with_eps(normalized_shape: usize, eps: f32) -> Self {
        Self {
            normalized_shape,
            eps,
        }
    }
}
