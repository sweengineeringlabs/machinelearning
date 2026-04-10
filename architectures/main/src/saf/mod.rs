// Re-export training-engine primitives (includes Trainer, Metrics, Checkpoint, etc.)
pub use training_engine::*;

// Models
pub use crate::core::models::lstm::LSTM;
pub use crate::core::models::lstm_forecast::LSTMForecast;
pub use crate::core::models::nbeats::NBeats;
pub use crate::core::models::tcn::TCN;
pub use crate::core::models::transformer::TimeSeriesTransformer;

// Data pipeline (domain-specific)
pub use crate::core::data::candle::OHLCVCandle;
pub use crate::core::data::dataset::{TimeSeriesDataset, TargetColumn};
pub use crate::core::data::dataloader::DataLoader;
pub use crate::core::data::features::{Feature, FeatureEngineer, Returns, MovingAverage, Volatility, RSI};
pub use crate::core::data::scaler::{Scaler, ScalerType};
