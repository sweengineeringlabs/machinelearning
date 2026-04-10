// Re-export from api — Storage is defined in api/storage_def.rs
pub use crate::api::storage_def::Storage;
pub(crate) use crate::api::storage_def::storage_byte_len;

/// Namespace marker for Storage re-export from this module.
pub(crate) struct StorageOps;
