pub mod config_ops;
pub mod device;
pub mod dtype;
pub mod error;
pub mod opt_profile_def;
pub mod pool_ops;
pub mod quant;
pub mod quant_config_def;
pub mod quant_config_builder_def;
pub mod shape;
pub mod storage_def;
pub mod tensor;
pub mod tensor_builder_def;
pub mod traits;

// Backward-compatible re-export modules so existing `crate::api::tensor_def::X` paths compile.
pub mod tensor_def {
    pub use crate::api::tensor::def::*;
    pub use crate::api::storage_def::*;
}

// Backward-compatible re-export so `crate::api::tensor_ops::TensorOps` still resolves.
pub mod tensor_ops {
    pub use crate::api::tensor::ops::*;
}

// Backward-compatible re-export so `crate::api::quant_ops::QuantOps` still resolves.
pub mod quant_ops {
    pub use crate::api::quant::ops::*;
}

// Backward-compatible re-export so `crate::api::quant_target_api::QuantTarget` still resolves.
pub mod quant_target_api {
    pub use crate::api::quant::target::*;
}
