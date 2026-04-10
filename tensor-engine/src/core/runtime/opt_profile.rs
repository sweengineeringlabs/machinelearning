use super::runtime_config::RuntimeConfig;

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
    /// Build a `RuntimeConfig` matching this profile.
    pub fn runtime_config(&self) -> RuntimeConfig {
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

    #[test]
    fn test_opt_profile_optimized_returns_default_thresholds() {
        let p = OptProfile::Optimized;
        let cfg = p.runtime_config();
        assert_eq!(cfg.softmax_par_threshold, 4096);
        assert_eq!(cfg.batched_matmul_par_threshold, 4096);
        assert_eq!(cfg.gemv_par_threshold, 4096);
        assert!(p.use_inplace_ops());
        assert!(p.use_buffered_sampling());
    }

    #[test]
    fn test_opt_profile_baseline_disables_parallelism() {
        let p = OptProfile::Baseline;
        let cfg = p.runtime_config();
        assert_eq!(cfg.softmax_par_threshold, usize::MAX);
        assert_eq!(cfg.batched_matmul_par_threshold, usize::MAX);
        assert_eq!(cfg.gemv_par_threshold, usize::MAX);
        assert!(!p.use_inplace_ops());
        assert!(!p.use_buffered_sampling());
    }

    #[test]
    fn test_opt_profile_aggressive_lowers_thresholds() {
        let p = OptProfile::Aggressive;
        let cfg = p.runtime_config();
        assert_eq!(cfg.softmax_par_threshold, 1024);
        assert_eq!(cfg.batched_matmul_par_threshold, 1024);
        assert_eq!(cfg.gemv_par_threshold, 1024);
        assert!(p.use_inplace_ops());
        assert!(p.use_buffered_sampling());
    }
}
