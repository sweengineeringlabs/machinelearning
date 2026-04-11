// Re-export training-engine primitives (includes Trainer, Metrics, Checkpoint, etc.)
pub use swe_ml_training::*;

// Re-export the architectures public API via api module
pub use crate::api::candle::OHLCVCandle;
pub use crate::api::feature::Feature;

// Models (via api facade)
pub use crate::api::models::LSTM;
pub use crate::api::models::LSTMBuilder;
pub use crate::api::models::LSTMForecast;
pub use crate::api::models::NBeats;
pub use crate::api::models::TCN;
pub use crate::api::models::TimeSeriesTransformer;

// Pipeline (via api facade)
pub use crate::api::pipeline::TimeSeriesDataset;
pub use crate::api::pipeline::TargetColumn;
pub use crate::api::pipeline::FeatureEngineer;
pub use crate::api::pipeline::Returns;
pub use crate::api::pipeline::MovingAverage;
pub use crate::api::pipeline::Volatility;
pub use crate::api::pipeline::RSI;
