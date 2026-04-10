//! # Tensor Engine
//!
//! Shared tensor engine for the machinelearning platform.
//!
//! Multi-dtype tensors with SIMD-accelerated operations, memory-mapped storage,
//! and configurable runtime profiles.
//!
//! ## Example
//!
//! ```rust
//! use tensor_engine::Tensor;
//!
//! let a = Tensor::randn([2, 3]);
//! let b = Tensor::randn([3, 4]);
//! let c = a.matmul(&b).unwrap();
//! assert_eq!(c.shape(), &[2, 4]);
//! ```

mod api;
mod core;
mod saf;
mod gateway;

pub use gateway::*;
