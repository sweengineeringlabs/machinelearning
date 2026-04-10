// SAF (Simple API Facade) -- re-exports for convenient access.
// All re-exports come from the api layer (not directly from core).

// API traits and types
pub use crate::api::error::{SwetsError, SwetsResult};
pub use crate::api::layer::Layer;
pub use crate::api::loss::Loss;
pub use crate::api::optim::{LRScheduler, Optimizer};
pub use crate::api::pool;
pub use crate::api::tape;
pub use crate::api::tensor::{Tensor, TensorId};

// Loss functions
pub use crate::api::CrossEntropyLoss;
pub use crate::api::HuberLoss;
pub use crate::api::MAELoss;
pub use crate::api::MSELoss;
pub use crate::api::QuantileLoss;

// Neural network layers
pub use crate::api::{GELU, ReLU, SiLU, Sigmoid, Tanh};
pub use crate::api::{BatchNorm1d, BatchNorm1dBuilder};
pub use crate::api::{Conv1d, Conv1dBuilder};
pub use crate::api::Dropout;
pub use crate::api::LayerNorm;
pub use crate::api::Linear;
pub use crate::api::Sequential;

// Optimizers
pub use crate::api::Adam;
pub use crate::api::AdamW;
pub use crate::api::{clip_grad_norm, clip_grad_value};
pub use crate::api::SGD;

// LR Schedulers
pub use crate::api::{CosineAnnealingLR, StepLR, WarmupCosineScheduler};

// Pipeline
pub use crate::api::dataset::Dataset;
pub use crate::api::{DataLoader, Scaler, ScalerType};

// Training infrastructure
pub use crate::api::Trainer;
pub use crate::api::Metrics;
pub use crate::api::model_summary;

// Serialization
pub use crate::api::{save_checkpoint, load_checkpoint, Checkpoint};

// Backward ops (for downstream crate tape-recorded operations)
pub use crate::api::{unbroadcast, AddBackward, MatMulBackward, MulBackward, ReLUBackward};
