pub mod dataset;
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

// Pipeline
pub use crate::core::pipeline::dataloader::DataLoader;
pub use crate::core::pipeline::scaler::{Scaler, ScalerType};

// Loss functions
pub use crate::core::lossfunction::cross_entropy::CrossEntropyLoss;
pub use crate::core::lossfunction::huber::HuberLoss;
pub use crate::core::lossfunction::mae_loss::MAELoss;
pub use crate::core::lossfunction::mse_loss::MSELoss;
pub use crate::core::lossfunction::quantile::QuantileLoss;

// Neural network layers
pub use crate::core::layers::activations::{GELU, ReLU, SiLU, Sigmoid, Tanh};
pub use crate::core::layers::batch_norm::BatchNorm1d;
pub use crate::core::layers::batch_norm_builder::BatchNorm1dBuilder;
pub use crate::core::layers::conv1d::Conv1d;
pub use crate::core::layers::conv1d_builder::Conv1dBuilder;
pub use crate::core::layers::dropout::Dropout;
pub use crate::core::layers::layer_norm::LayerNorm;
pub use crate::core::layers::linear::Linear;
pub use crate::core::layers::sequential::Sequential;

// Optimizers
pub use crate::core::optimizer::adam::Adam;
pub use crate::core::optimizer::adamw::AdamW;
pub use crate::core::optimizer::grad_clip::{clip_grad_norm, clip_grad_value};
pub use crate::core::optimizer::sgd::SGD;

// LR Schedulers
pub use crate::core::optimizer::schedulers::{CosineAnnealingLR, StepLR, WarmupCosineScheduler};

// Training infrastructure
pub use crate::core::runner::trainer::Trainer;
pub use crate::core::runner::metrics::Metrics;
pub use crate::core::runner::summary::model_summary;

// Serialization
pub use crate::core::checkpoint::{save_checkpoint, load_checkpoint, Checkpoint};

// Backward ops (for downstream crate tape-recorded operations)
pub use crate::core::gradient::add::{unbroadcast, AddBackward};
pub use crate::core::gradient::matmul::MatMulBackward;
pub use crate::core::gradient::mul::MulBackward;
pub use crate::core::gradient::relu::ReLUBackward;
