//! # swe-llmmodel-io
//!
//! Tensor format I/O. Reads SafeTensors (zero-copy mmap) and reads/writes
//! a custom CRC32-checked binary format.

pub mod api;
mod core;
mod saf;

pub use saf::*;
