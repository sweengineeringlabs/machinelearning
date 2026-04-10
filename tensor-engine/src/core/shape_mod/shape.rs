// Re-export from api — Shape is a data type, defined in api/
pub use crate::api::shape::{self, Shape};

/// Namespace marker for Shape operations implemented in this module.
pub(crate) struct ShapeOps;
