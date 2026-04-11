/// Contract for thread pool configuration.
///
/// Controls parallelism thresholds for matmul, softmax, and other
/// parallel operations. Different configurations suit different
/// hardware (high core count vs few fast cores).
pub trait ThreadConfig: Send + Sync {
    /// Number of rayon threads to use.
    fn num_threads(&self) -> usize;

    /// Minimum tensor dimension for parallel matmul.
    fn matmul_parallel_threshold(&self) -> usize;

    /// Minimum number of elements for parallel softmax.
    fn softmax_parallel_threshold(&self) -> usize;

    /// Apply this configuration to the runtime.
    fn apply(&self) -> Result<(), String>;

    /// Returns a human-readable description.
    fn describe(&self) -> &str;
}
