use crate::api::traits::ComputeBackend;
use swe_ml_tensor::Tensor;

/// CPU compute backend — scalar + SIMD (AVX2/NEON).
///
/// This is the current default. Uses swe-ml-tensor's built-in
/// matmul, softmax, and activation functions.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
        a.matmul(b).map_err(|e| e.to_string())
    }

    fn softmax(&self, input: &Tensor) -> Result<Tensor, String> {
        input.softmax(-1).map_err(|e| e.to_string())
    }

    fn gelu(&self, input: &Tensor) -> Result<Tensor, String> {
        Ok(input.gelu())
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor, String> {
        Ok(input.silu())
    }

    fn name(&self) -> &str {
        if cfg!(target_feature = "avx2") {
            "cpu-avx2"
        } else if cfg!(target_arch = "aarch64") {
            "cpu-neon"
        } else {
            "cpu-scalar"
        }
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }
}

/// Vulkan compute backend — GPU acceleration via compute shaders.
///
/// Not yet implemented. When complete, will provide 10-50x speedup
/// for matmul-bound workloads on any GPU (NVIDIA, AMD, Intel).
pub struct VulkanBackend;

impl ComputeBackend for VulkanBackend {
    fn matmul(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor, String> {
        Err("Vulkan backend not yet implemented".into())
    }

    fn softmax(&self, _input: &Tensor) -> Result<Tensor, String> {
        Err("Vulkan backend not yet implemented".into())
    }

    fn gelu(&self, _input: &Tensor) -> Result<Tensor, String> {
        Err("Vulkan backend not yet implemented".into())
    }

    fn silu(&self, _input: &Tensor) -> Result<Tensor, String> {
        Err("Vulkan backend not yet implemented".into())
    }

    fn name(&self) -> &str {
        "vulkan"
    }

    fn is_available(&self) -> bool {
        false // Not yet implemented
    }
}
