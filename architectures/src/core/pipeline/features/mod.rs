/// Feature engineering for OHLCV time series data.
///
/// FR-705: Pluggable feature trait and engine.
/// FR-706: Built-in technical indicators (Returns, SMA, Volatility).
/// FR-707: RSI indicator using Wilder's EMA.

pub mod feature_engineer;
pub mod returns;
pub mod moving_average;
pub mod volatility;
pub mod rsi;

pub use crate::api::feature::Feature;
pub use feature_engineer::FeatureEngineer;
pub use returns::Returns;
pub use moving_average::MovingAverage;
pub use volatility::Volatility;
pub use rsi::RSI;
