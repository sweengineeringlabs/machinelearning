use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;

/// Global threshold for switching softmax from sequential to parallel (rayon).
pub(crate) static SOFTMAX_PAR_THRESHOLD: AtomicUsize = AtomicUsize::new(4096);

/// Global threshold for switching batched_matmul from sequential to parallel (rayon).
pub(crate) static BATCHED_MATMUL_PAR_THRESHOLD: AtomicUsize = AtomicUsize::new(4096);

/// Global threshold for switching F32 gemv (M=1) to custom parallel path.
/// When N >= this threshold, use parallel gemv instead of faer.
pub(crate) static GEMV_PAR_THRESHOLD: AtomicUsize = AtomicUsize::new(4096);

/// Runtime configuration for parallelism and thread management.
/// Must be applied (via `apply()`) before any computation to take effect.
pub struct RuntimeConfig {
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

impl RuntimeConfig {
    /// Apply this runtime configuration globally.
    ///
    /// Sets faer's global parallelism and optionally configures
    /// rayon's global thread pool. Writes optimization thresholds
    /// to global atomics. Must be called before any computation
    /// (matmul, attention, etc.) for settings to take effect.
    pub fn apply(&self) -> Result<(), crate::api::error::TensorError> {
        use faer::{Parallelism, set_global_parallelism};

        if self.num_threads == 0 {
            set_global_parallelism(Parallelism::Rayon(0));
        } else {
            set_global_parallelism(Parallelism::Rayon(self.num_threads));
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_threads)
                .build_global()
                .map_err(|e| crate::api::error::TensorError::InvalidOperation(
                    format!("Failed to set rayon thread pool: {}", e)
                ))?;
        }

        // Write optimization thresholds to global atomics
        SOFTMAX_PAR_THRESHOLD.store(self.softmax_par_threshold, Ordering::Relaxed);
        BATCHED_MATMUL_PAR_THRESHOLD.store(self.batched_matmul_par_threshold, Ordering::Relaxed);
        GEMV_PAR_THRESHOLD.store(self.gemv_par_threshold, Ordering::Relaxed);

        // Log SIMD capabilities
        let simd = Self::detect_simd();
        eprintln!("[runtime] SIMD: {}", simd);

        // Log thread count
        let threads = rayon::current_num_threads();
        eprintln!("[runtime] Rayon threads: {}", threads);

        Ok(())
    }

    /// Detect available SIMD instruction sets.
    pub fn detect_simd() -> &'static str {
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
            // NEON is always available on aarch64
            return "NEON";
        }
        "scalar"
    }

    /// Warm up the rayon thread pool by forcing all threads to wake and do work.
    ///
    /// This reduces jitter on the first few parallel operations by:
    /// 1. Spawning all threads in the pool (if not already spawned)
    /// 2. Warming up each thread's instruction cache with SIMD code paths
    /// 3. Touching memory to populate TLB entries
    ///
    /// Call this after model loading but before timed inference.
    pub fn warmup_thread_pool() {
        let n_threads = rayon::current_num_threads();

        // Allocate enough work to ensure every thread gets some
        // Use a larger buffer to exercise memory subsystem
        let work_size = n_threads * 1024; // 1K elements per thread
        let mut buffer: Vec<f32> = vec![1.0; work_size];

        // Force parallel iteration that touches every element
        // This wakes all threads and warms their caches
        buffer.par_chunks_mut(1024).for_each(|chunk| {
            // Do enough work to warm instruction cache (SIMD paths)
            for i in 0..chunk.len() {
                chunk[i] = chunk[i] * 2.0 + 1.0;
            }
            // Memory barrier to ensure writes complete
            std::sync::atomic::fence(Ordering::SeqCst);
        });

        // Second pass: simulate matmul-like access pattern
        // Each thread reads from different offsets (like column-parallel matmul)
        let _sum: f32 = buffer.par_chunks(1024)
            .map(|chunk| chunk.iter().sum::<f32>())
            .sum();
    }
}

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

/// Per-layer-type quantization target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantTarget {
    /// Keep in F32 (no quantization).
    None,
    /// Half-precision float16 — lossless for BF16 models, 2 bytes/param.
    F16,
    /// Block-quantized 8-bit (default for most layers).
    Q8_0,
    /// Block-quantized 4-bit (aggressive compression).
    Q4_0,
    /// Block-quantized 4-bit with min offset (better quality than Q4_0).
    Q4_1,
}

/// Quantization strategy: per-layer-type policies loaded from TOML.
///
/// Each field controls how a class of weights is quantized after loading.
/// `None` = keep original dtype, `Q8_0` / `Q4_0` = quantize.
///
/// ```toml
/// [quantization]
/// attention = "q8_0"
/// feed_forward = "q8_0"
/// output = "none"
/// gate = "none"          # PLE gates
/// min_dim = 1536
/// ```
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct QuantStrategy {
    /// Attention Q/K/V/O projections.
    pub attention: QuantTarget,
    /// Feed-forward up/gate/down projections.
    pub feed_forward: QuantTarget,
    /// Output (lm_head) projection.
    pub output: QuantTarget,
    /// MoE expert projections.
    pub moe: QuantTarget,
    /// PLE gate projections.
    pub gate: QuantTarget,
    /// Minimum dimension to quantize (skip smaller layers).
    pub min_dim: usize,
}

impl Default for QuantStrategy {
    fn default() -> Self {
        Self {
            attention: QuantTarget::Q8_0,
            feed_forward: QuantTarget::Q8_0,
            output: QuantTarget::Q8_0,
            moe: QuantTarget::Q8_0,
            gate: QuantTarget::Q8_0,
            min_dim: 0,
        }
    }
}

impl QuantStrategy {
    /// All layers in F32 — no quantization.
    pub fn none() -> Self {
        Self {
            attention: QuantTarget::None,
            feed_forward: QuantTarget::None,
            output: QuantTarget::None,
            moe: QuantTarget::None,
            gate: QuantTarget::None,
            min_dim: 0,
        }
    }

    /// Default Q8_0 for all layers (current behavior).
    pub fn q8_all() -> Self {
        Self::default()
    }

    /// Conservative: keep output in F32, quantize the rest.
    /// Good for models with logit softcapping (Gemma 4).
    pub fn q8_preserve_output() -> Self {
        Self {
            output: QuantTarget::None,
            ..Self::default()
        }
    }

    /// All layers in F16 — half memory of F32, no quantization noise.
    /// ~same size as original BF16 weights.
    pub fn f16_all() -> Self {
        Self {
            attention: QuantTarget::F16,
            feed_forward: QuantTarget::F16,
            output: QuantTarget::F16,
            moe: QuantTarget::F16,
            gate: QuantTarget::F16,
            min_dim: 0,
        }
    }

    /// Load from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        struct Wrapper {
            #[serde(default)]
            quantization: QuantStrategy,
        }
        let wrapper: Wrapper = toml::from_str(toml_str)
            .map_err(|e| format!("Failed to parse quantization config: {}", e))?;
        Ok(wrapper.quantization)
    }

    /// Load from a TOML file path. Falls back to default if file doesn't exist.
    pub fn from_toml_file(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => Self::from_toml(&contents).unwrap_or_else(|e| {
                log::warn!("Invalid quantization config {}: {}. Using default.", path.display(), e);
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let config = RuntimeConfig::default();
        assert_eq!(config.num_threads, 0);
        assert_eq!(config.softmax_par_threshold, 4096);
        assert_eq!(config.batched_matmul_par_threshold, 4096);
        assert_eq!(config.gemv_par_threshold, 4096);
    }

    #[test]
    fn test_detect_simd() {
        let simd = RuntimeConfig::detect_simd();
        assert!(!simd.is_empty());
    }

    #[test]
    fn test_opt_profile_optimized() {
        let p = OptProfile::Optimized;
        let cfg = p.runtime_config();
        assert_eq!(cfg.softmax_par_threshold, 4096);
        assert_eq!(cfg.batched_matmul_par_threshold, 4096);
        assert_eq!(cfg.gemv_par_threshold, 4096);
        assert!(p.use_inplace_ops());
        assert!(p.use_buffered_sampling());
    }

    #[test]
    fn test_opt_profile_baseline() {
        let p = OptProfile::Baseline;
        let cfg = p.runtime_config();
        assert_eq!(cfg.softmax_par_threshold, usize::MAX);
        assert_eq!(cfg.batched_matmul_par_threshold, usize::MAX);
        assert_eq!(cfg.gemv_par_threshold, usize::MAX);
        assert!(!p.use_inplace_ops());
        assert!(!p.use_buffered_sampling());
    }

    #[test]
    fn test_opt_profile_aggressive() {
        let p = OptProfile::Aggressive;
        let cfg = p.runtime_config();
        assert_eq!(cfg.softmax_par_threshold, 1024);
        assert_eq!(cfg.batched_matmul_par_threshold, 1024);
        assert_eq!(cfg.gemv_par_threshold, 1024);
        assert!(p.use_inplace_ops());
        assert!(p.use_buffered_sampling());
    }

    #[test]
    fn test_quant_strategy_default_is_q8_all() {
        let s = QuantStrategy::default();
        assert_eq!(s.attention, QuantTarget::Q8_0);
        assert_eq!(s.feed_forward, QuantTarget::Q8_0);
        assert_eq!(s.output, QuantTarget::Q8_0);
    }

    #[test]
    fn test_quant_strategy_none_skips_all() {
        let s = QuantStrategy::none();
        assert_eq!(s.attention, QuantTarget::None);
        assert_eq!(s.feed_forward, QuantTarget::None);
        assert_eq!(s.output, QuantTarget::None);
        assert_eq!(s.moe, QuantTarget::None);
    }

    #[test]
    fn test_quant_strategy_from_toml() {
        let toml = r#"
[quantization]
attention = "q8_0"
feed_forward = "q8_0"
output = "none"
gate = "none"
min_dim = 1536
"#;
        let s = QuantStrategy::from_toml(toml).unwrap();
        assert_eq!(s.attention, QuantTarget::Q8_0);
        assert_eq!(s.output, QuantTarget::None);
        assert_eq!(s.gate, QuantTarget::None);
        assert_eq!(s.min_dim, 1536);
    }

    #[test]
    fn test_quant_strategy_from_toml_partial() {
        // Only override output; rest use defaults
        let toml = r#"
[quantization]
output = "none"
"#;
        let s = QuantStrategy::from_toml(toml).unwrap();
        assert_eq!(s.attention, QuantTarget::Q8_0); // default
        assert_eq!(s.output, QuantTarget::None);    // overridden
    }

    #[test]
    fn test_quant_strategy_from_toml_empty_uses_defaults() {
        let s = QuantStrategy::from_toml("").unwrap();
        assert_eq!(s.attention, QuantTarget::Q8_0);
        assert_eq!(s.output, QuantTarget::Q8_0);
    }
}
