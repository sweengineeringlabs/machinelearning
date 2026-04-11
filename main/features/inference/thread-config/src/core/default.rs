use crate::api::traits::ThreadConfig;

/// Auto-detected thread configuration.
///
/// Uses all available CPU cores and applies default parallelism thresholds
/// tuned from benchmarking on typical hardware.
pub struct AutoThreadConfig {
    num_threads: usize,
}

impl AutoThreadConfig {
    pub fn new() -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }
}

impl Default for AutoThreadConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadConfig for AutoThreadConfig {
    fn num_threads(&self) -> usize {
        self.num_threads
    }

    fn matmul_parallel_threshold(&self) -> usize {
        8192 // elements — below this, sequential is faster
    }

    fn softmax_parallel_threshold(&self) -> usize {
        4096
    }

    fn apply(&self) -> Result<(), String> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build_global()
            .map_err(|e| e.to_string())
    }

    fn describe(&self) -> &str {
        "Auto-detected thread configuration"
    }
}
