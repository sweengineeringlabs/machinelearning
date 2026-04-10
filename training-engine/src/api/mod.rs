pub mod error;
pub mod layer;
pub mod loss;
pub mod lr_scheduler;
pub mod optim;
pub mod optimizer;
pub mod pool;
pub mod tape;
pub mod tensor;
pub mod tensor_id;
pub mod traits;

// Re-export concrete types from core for the SAF layer to consume.

// Loss functions
pub use crate::core::loss::cross_entropy::CrossEntropyLoss;
pub use crate::core::loss::huber::HuberLoss;
pub use crate::core::loss::mae_loss::MAELoss;
pub use crate::core::loss::mse_loss::MSELoss;
pub use crate::core::loss::quantile::QuantileLoss;

// Neural network layers
pub use crate::core::nn::activations::{GELU, ReLU, SiLU, Sigmoid, Tanh};
pub use crate::core::nn::batch_norm::BatchNorm1d;
pub use crate::core::nn::batch_norm_builder::BatchNorm1dBuilder;
pub use crate::core::nn::conv1d::Conv1d;
pub use crate::core::nn::conv1d_builder::Conv1dBuilder;
pub use crate::core::nn::dropout::Dropout;
pub use crate::core::nn::layer_norm::LayerNorm;
pub use crate::core::nn::linear::Linear;
pub use crate::core::nn::sequential::Sequential;

// Optimizers
pub use crate::core::optim::adam::Adam;
pub use crate::core::optim::adamw::AdamW;
pub use crate::core::optim::grad_clip::{clip_grad_norm, clip_grad_value};
pub use crate::core::optim::sgd::SGD;

// LR Schedulers
pub use crate::core::optim::schedulers::{CosineAnnealingLR, StepLR, WarmupCosineScheduler};

// Training infrastructure
pub use crate::core::training::trainer::Trainer;
pub use crate::core::training::metrics::Metrics;
pub use crate::core::training::summary::model_summary;

// Serialization
pub use crate::core::serde::{save_checkpoint, load_checkpoint, Checkpoint};

// Backward ops (for downstream crate tape-recorded operations)
pub use crate::core::ops::add::{unbroadcast, AddBackward};
pub use crate::core::ops::matmul::MatMulBackward;
pub use crate::core::ops::mul::MulBackward;
pub use crate::core::ops::relu::ReLUBackward;
