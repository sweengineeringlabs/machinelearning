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
