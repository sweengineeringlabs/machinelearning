use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;
use crate::api::traits::ConfigOps;

/// Global threshold for switching softmax from sequential to parallel (rayon).
pub(crate) static SOFTMAX_PAR_THRESHOLD: AtomicUsize = AtomicUsize::new(4096);

/// Global threshold for switching batched_matmul from sequential to parallel (rayon).
pub(crate) static BATCHED_MATMUL_PAR_THRESHOLD: AtomicUsize = AtomicUsize::new(4096);

/// Global threshold for switching F32 gemv (M=1) to custom parallel path.
pub(crate) static GEMV_PAR_THRESHOLD: AtomicUsize = AtomicUsize::new(4096);

/// Runtime configuration for parallelism and thread management.
/// Must be applied (via `apply()`) before any computation to take effect.
pub(crate) struct RuntimeConfig {
    /// Number of threads for faer and rayon parallelism.
    /// 0 means auto-detect (use all available cores).
    pub num_threads: usize,
    /// Element count below which softmax uses a sequential path (default 4096).
    pub softmax_par_threshold: usize,
    /// Element count below which batched_matmul uses a sequential path (default 4096).
    pub batched_matmul_par_threshold: usize,
    /// Minimum N (out_features) for parallel F32 gemv when M=1 (default 4096).
    pub gemv_par_threshold: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,
            softmax_par_threshold: 4096,
            batched_matmul_par_threshold: 4096,
            gemv_par_threshold: 4096,
        }
    }
}

/// Apply a runtime configuration globally.
pub(crate) fn apply_runtime_config(config: &RuntimeConfig) -> Result<(), crate::api::error::TensorError> {
    use faer::{Parallelism, set_global_parallelism};

    if config.num_threads == 0 {
        set_global_parallelism(Parallelism::Rayon(0));
    } else {
        set_global_parallelism(Parallelism::Rayon(config.num_threads));
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build_global()
            .map_err(|e| crate::api::error::TensorError::InvalidOperation(
                format!("Failed to set rayon thread pool: {}", e)
            ))?;
    }

    SOFTMAX_PAR_THRESHOLD.store(config.softmax_par_threshold, Ordering::Relaxed);
    BATCHED_MATMUL_PAR_THRESHOLD.store(config.batched_matmul_par_threshold, Ordering::Relaxed);
    GEMV_PAR_THRESHOLD.store(config.gemv_par_threshold, Ordering::Relaxed);

    let simd = detect_simd();
    eprintln!("[runtime] SIMD: {}", simd);

    let threads = rayon::current_num_threads();
    eprintln!("[runtime] Rayon threads: {}", threads);

    Ok(())
}

/// Detect available SIMD instruction sets.
pub(crate) fn detect_simd() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
        if is_x86_feature_detected!("sse2") {
            return "SSE2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON";
    }
    "scalar"
}

/// Warm up the rayon thread pool by forcing all threads to wake and do work.
pub(crate) fn warmup_thread_pool() {
    let n_threads = rayon::current_num_threads();
    let work_size = n_threads * 1024;
    let mut buffer: Vec<f32> = vec![1.0; work_size];

    buffer.par_chunks_mut(1024).for_each(|chunk| {
        for i in 0..chunk.len() {
            chunk[i] = chunk[i] * 2.0 + 1.0;
        }
        std::sync::atomic::fence(Ordering::SeqCst);
    });

    let _sum: f32 = buffer.par_chunks(1024)
        .map(|chunk| chunk.iter().sum::<f32>())
        .sum();
}

impl ConfigOps for RuntimeConfig {
    fn apply(&self) -> Result<(), crate::api::error::TensorError> {
        apply_runtime_config(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: RuntimeConfig (Default)
    #[test]
    fn test_default_returns_standard_thresholds() {
        let config = RuntimeConfig::default();
        assert_eq!(config.num_threads, 0);
        assert_eq!(config.softmax_par_threshold, 4096);
        assert_eq!(config.batched_matmul_par_threshold, 4096);
        assert_eq!(config.gemv_par_threshold, 4096);
    }

    /// @covers: detect_simd
    #[test]
    fn test_detect_simd_returns_nonempty() {
        let simd = detect_simd();
        assert!(!simd.is_empty());
    }

    /// @covers: apply_runtime_config
    #[test]
    fn test_apply_runtime_config_does_not_panic() {
        let config = RuntimeConfig::default();
        let result = apply_runtime_config(&config);
        assert!(result.is_ok());
    }

    /// @covers: warmup_thread_pool
    #[test]
    fn test_warmup_thread_pool_does_not_panic() {
        warmup_thread_pool();
    }

    /// @covers: ConfigOps::apply
    #[test]
    fn test_apply_trait_delegates_to_inner() {
        let config = RuntimeConfig::default();
        let result = config.apply();
        assert!(result.is_ok());
    }
}
