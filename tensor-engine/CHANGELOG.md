# Changelog

## [0.1.0] - 2026-04-10

### Added
- Multi-dtype Tensor (F32, F16, BF16, I8, U8, Q8_0, Q4_0, Q4_1)
- SIMD-accelerated ops: matmul, softmax, RMSNorm (AVX2/NEON)
- Memory-mapped file-backed storage for large models
- SmallVec shape optimization for tensors up to 4D
- Broadcasting support for element-wise operations
- Arena allocator for batch memory management
- RuntimeConfig with parallelism thresholds and optimization profiles
- QuantStrategy with TOML-configurable per-layer quantization targeting
