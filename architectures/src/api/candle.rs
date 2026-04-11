/// OHLCV candle data point representing a single time period.
///
/// FR-700: Core data structure for financial time series data.
#[derive(Debug, Clone)]
pub struct OHLCVCandle {
    /// Unix timestamp (seconds since epoch).
    pub timestamp: i64,
    /// Opening price.
    pub open: f32,
    /// Highest price during the period.
    pub high: f32,
    /// Lowest price during the period.
    pub low: f32,
    /// Closing price.
    pub close: f32,
    /// Volume traded during the period.
    pub volume: f32,
}

impl OHLCVCandle {
    /// Create a new OHLCV candle.
    pub fn new(timestamp: i64, open: f32, high: f32, low: f32, close: f32, volume: f32) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Extract a feature value by column name.
    ///
    /// Returns `None` if the column name is not recognized.
    pub fn get_feature(&self, name: &str) -> Option<f32> {
        match name {
            "open" => Some(self.open),
            "high" => Some(self.high),
            "low" => Some(self.low),
            "close" => Some(self.close),
            "volume" => Some(self.volume),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: get_feature
    #[test]
    fn test_new_creates_candle() {
        let candle = OHLCVCandle::new(1000, 10.0, 12.0, 9.0, 11.0, 500.0);
        assert_eq!(candle.timestamp, 1000);
        assert_eq!(candle.open, 10.0);
        assert_eq!(candle.close, 11.0);
    }

    /// @covers: get_feature
    #[test]
    fn test_get_feature_returns_none_for_unknown() {
        let candle = OHLCVCandle::new(0, 1.0, 2.0, 0.5, 1.5, 100.0);
        assert_eq!(candle.get_feature("nonexistent"), None);
    }

    /// @covers: get_feature
    #[test]
    fn test_get_feature_returns_correct_values() {
        let candle = OHLCVCandle::new(0, 1.0, 2.0, 0.5, 1.5, 100.0);
        assert_eq!(candle.get_feature("open"), Some(1.0));
        assert_eq!(candle.get_feature("high"), Some(2.0));
        assert_eq!(candle.get_feature("low"), Some(0.5));
        assert_eq!(candle.get_feature("close"), Some(1.5));
        assert_eq!(candle.get_feature("volume"), Some(100.0));
    }
}
