use swe_ml_tensor::Tensor;

/// Contract for compute backends.
///
/// Abstracts the hardware that runs matrix operations:
///
/// - **Cpu**: Current default — scalar + SIMD (AVX2/NEON)
/// - **Vulkan**: GPU compute shaders — 10-50x speedup for matmul
/// - **Metal**: macOS GPU acceleration
/// - **Cuda**: NVIDIA GPU acceleration
pub trait ComputeBackend: Send + Sync {
    /// Matrix multiply: C = A × B
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, String>;

    /// Softmax along the last dimension.
    fn softmax(&self, input: &Tensor) -> Result<Tensor, String>;

    /// Element-wise GELU activation.
    fn gelu(&self, input: &Tensor) -> Result<Tensor, String>;

    /// Element-wise SiLU activation.
    fn silu(&self, input: &Tensor) -> Result<Tensor, String>;

    /// Returns the backend name (e.g., "cpu-avx2", "vulkan", "metal").
    fn name(&self) -> &str;

    /// Returns whether this backend is available on the current hardware.
    fn is_available(&self) -> bool;
}
