pub use crate::api::opt_profile_def::OptProfile;
use super::runtime_config::RuntimeConfig;

/// Namespace marker for OptProfile implementation methods defined in this module.
pub(crate) struct OptProfileImpl;

impl OptProfile {
    /// Apply this profile's runtime configuration globally.
    pub fn apply(&self) -> Result<(), crate::api::error::TensorError> {
        crate::core::runtime::runtime_config::apply_runtime_config(&self.runtime_config())
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

    /// @covers: apply
    #[test]
    fn test_apply_optimized_succeeds() {
        assert!(OptProfile::Optimized.apply().is_ok());
    }

    /// @covers: runtime_config
    #[test]
    fn test_runtime_config_optimized_has_default_thresholds() {
        let config = OptProfile::Optimized.runtime_config();
        assert_eq!(config.softmax_par_threshold, 4096);
    }

    /// @covers: runtime_config
    #[test]
    fn test_runtime_config_baseline_has_max_thresholds() {
        let config = OptProfile::Baseline.runtime_config();
        assert_eq!(config.softmax_par_threshold, usize::MAX);
    }

    /// @covers: runtime_config
    #[test]
    fn test_runtime_config_aggressive_has_low_thresholds() {
        let config = OptProfile::Aggressive.runtime_config();
        assert_eq!(config.softmax_par_threshold, 1024);
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
