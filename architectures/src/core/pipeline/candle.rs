// Re-export OHLCVCandle from the api layer (source of truth)
pub use crate::api::candle::OHLCVCandle;

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: new
    #[test]
    fn test_candle_construction() {
        let candle = OHLCVCandle::new(1_700_000_000, 100.0, 105.0, 99.0, 103.0, 50_000.0);
        assert_eq!(candle.timestamp, 1_700_000_000);
        assert_eq!(candle.open, 100.0);
        assert_eq!(candle.high, 105.0);
        assert_eq!(candle.low, 99.0);
        assert_eq!(candle.close, 103.0);
        assert_eq!(candle.volume, 50_000.0);
    }

    /// @covers: clone
    #[test]
    fn test_candle_clone() {
        let candle = OHLCVCandle::new(1_700_000_000, 100.0, 105.0, 99.0, 103.0, 50_000.0);
        let cloned = candle.clone();
        assert_eq!(candle.timestamp, cloned.timestamp);
        assert_eq!(candle.close, cloned.close);
    }

    /// @covers: get_feature
    #[test]
    fn test_get_feature() {
        let candle = OHLCVCandle::new(0, 10.0, 20.0, 5.0, 15.0, 1000.0);
        assert_eq!(candle.get_feature("open"), Some(10.0));
        assert_eq!(candle.get_feature("high"), Some(20.0));
        assert_eq!(candle.get_feature("low"), Some(5.0));
        assert_eq!(candle.get_feature("close"), Some(15.0));
        assert_eq!(candle.get_feature("volume"), Some(1000.0));
        assert_eq!(candle.get_feature("unknown"), None);
    }
}
