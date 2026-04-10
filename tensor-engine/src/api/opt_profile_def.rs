/// Optimization profiles for A/B benchmarking.
///
/// Controls rayon thresholds and whether in-place/buffered optimizations are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptProfile {
    /// All optimizations enabled (default behavior).
    Optimized,
    /// All optimizations disabled: allocating paths, thresholds set to MAX.
    Baseline,
    /// Lower thresholds (1024) for aggressive parallelism.
    Aggressive,
}

impl OptProfile {
    /// Whether in-place ops should be used (attention scaling, residual adds).
    pub fn use_inplace_ops(&self) -> bool {
        *self != OptProfile::Baseline
    }

    /// Whether buffered sampling should be used (pre-allocated logits/sort buffers).
    pub fn use_buffered_sampling(&self) -> bool {
        *self != OptProfile::Baseline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: use_inplace_ops
    #[test]
    fn test_use_inplace_ops_disabled_for_baseline() {
        assert!(!OptProfile::Baseline.use_inplace_ops());
        assert!(OptProfile::Optimized.use_inplace_ops());
        assert!(OptProfile::Aggressive.use_inplace_ops());
    }

    /// @covers: use_buffered_sampling
    #[test]
    fn test_use_buffered_sampling_disabled_for_baseline() {
        assert!(!OptProfile::Baseline.use_buffered_sampling());
        assert!(OptProfile::Optimized.use_buffered_sampling());
        assert!(OptProfile::Aggressive.use_buffered_sampling());
    }

    #[test]
    fn test_opt_profile_equality_comparison() {
        assert_eq!(OptProfile::Optimized, OptProfile::Optimized);
        assert_ne!(OptProfile::Optimized, OptProfile::Baseline);
        assert_ne!(OptProfile::Baseline, OptProfile::Aggressive);
    }

    #[test]
    fn test_opt_profile_clone_preserves_variant() {
        let p = OptProfile::Aggressive;
        let cloned = p.clone();
        assert_eq!(p, cloned);
    }
}
