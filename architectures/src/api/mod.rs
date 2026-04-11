// Re-export training-engine API for backwards compatibility.
// Since swe_ml_training::api is private, we re-create the module structure
// using the public surface.

pub mod error {
    pub use swe_ml_training::{SwetsError, SwetsResult};
}

pub mod candle;
pub mod feature;

pub mod layer {
    pub use swe_ml_training::Layer;
}

pub mod loss {
    pub use swe_ml_training::Loss;
}

pub mod optim {
    pub use swe_ml_training::{LRScheduler, Optimizer};
}

pub use swe_ml_training::pool;
pub use swe_ml_training::tape;

pub mod tensor {
    pub use swe_ml_training::{Tensor, TensorId};
}

pub mod traits;

// Re-export model types from core/ so saf/ can re-export from api/
pub mod models {
    pub use crate::core::models::lstm::LSTM;
    pub use crate::core::models::lstm::LSTMBuilder;
    pub use crate::core::models::lstm::forecast::LSTMForecast;
    pub use crate::core::models::nbeats::NBeats;
    pub use crate::core::models::tcn::TCN;
    pub use crate::core::models::transformer::TimeSeriesTransformer;
}

// Re-export pipeline types from core/ so saf/ can re-export from api/
pub mod pipeline {
    pub use crate::core::pipeline::dataset::TimeSeriesDataset;
    pub use crate::core::pipeline::target_column::TargetColumn;
    pub use crate::core::pipeline::features::{FeatureEngineer, Returns, MovingAverage, Volatility, RSI};
}
