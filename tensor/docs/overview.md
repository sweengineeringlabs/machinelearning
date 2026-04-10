# Tensor Engine

Shared tensor engine providing multi-dtype tensors with SIMD-accelerated operations.

## Supported DTypes

| DType | Size | Description |
|-------|------|-------------|
| F32 | 4 bytes | 32-bit float (default) |
| F16 | 2 bytes | 16-bit float |
| BF16 | 2 bytes | Brain floating point |
| I8 | 1 byte | Signed 8-bit integer |
| U8 | 1 byte | Unsigned 8-bit integer |
| Q8_0 | ~1 byte | 8-bit block quantized (32 elements/block) |
| Q4_0 | ~0.56 bytes | 4-bit block quantized |
| Q4_1 | ~0.63 bytes | 4-bit block quantized with min offset |

## Storage Backends

- **Owned** — heap-allocated `Vec<u8>`
- **View** — sliced reference to parent tensor
- **MMap** — memory-mapped file for large model weights

## SIMD Acceleration

AVX2 (x86) and NEON (ARM) kernels for:
- Matrix multiplication
- Softmax with parallel reductions
- RMSNorm with fused normalize-scale

## Architecture

```
tensor-engine/
├── api/          Public types (DType, Device, TensorError)
├── core/
│   ├── tensor/   Tensor struct, math ops, views
│   ├── shape.rs  Shape with strides, broadcasting
│   ├── arena.rs  Batch memory allocator
│   └── runtime.rs RuntimeConfig, QuantStrategy
└── saf/          Facade re-exports
```
