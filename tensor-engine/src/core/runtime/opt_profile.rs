pub use crate::api::opt_profile_def::OptProfile;
use super::runtime_config::RuntimeConfig;

impl OptProfile {
    /// Apply this profile's runtime configuration globally.
    pub fn apply(&self) -> Result<(), crate::api::error::TensorError> {
        self.runtime_config().apply_inner()
    }

    /// Build a `RuntimeConfig` matching this profile.
    pub(crate) fn runtime_config(&self) -> RuntimeConfig {
        match self {
            OptProfile::Optimized => RuntimeConfig::default(),
            OptProfile::Baseline => RuntimeConfig {
                softmax_par_threshold: usize::MAX,
                batched_matmul_par_threshold: usize::MAX,
                gemv_par_threshold: usize::MAX,
                ..RuntimeConfig::default()
            },
            OptProfile::Aggressive => RuntimeConfig {
                softmax_par_threshold: 1024,
                batched_matmul_par_threshold: 1024,
                gemv_par_threshold: 1024,
                ..RuntimeConfig::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: OptProfile::apply
    #[test]
    fn test_opt_profile_optimized_applies_without_error() {
        assert!(OptProfile::Optimized.apply().is_ok());
    }

    /// @covers: OptProfile::use_inplace_ops
    #[test]
    fn test_opt_profile_baseline_disables_inplace() {
        assert!(!OptProfile::Baseline.use_inplace_ops());
        assert!(OptProfile::Optimized.use_inplace_ops());
    }

    /// @covers: OptProfile::use_buffered_sampling
    #[test]
    fn test_opt_profile_aggressive_enables_buffered_sampling() {
        assert!(OptProfile::Aggressive.use_buffered_sampling());
        assert!(!OptProfile::Baseline.use_buffered_sampling());
    }
}
