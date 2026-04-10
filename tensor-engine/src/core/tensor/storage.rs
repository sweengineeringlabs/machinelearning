// Re-export from api — Storage is defined alongside Tensor in api/tensor_def.rs
pub use crate::api::tensor_def::Storage;
pub(crate) use crate::api::tensor_def::storage_byte_len;
