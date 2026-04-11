use super::candle::OHLCVCandle;

/// A computed feature derived from OHLCV candle data.
pub trait Feature: Send + Sync {
    /// Compute the feature for each candle in `data`.
    ///
    /// The returned vector has the same length as `data`. Where the feature
    /// is undefined (e.g., first element for returns) it should be `0.0` or
    /// another sensible default.
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32>;

    /// Human-readable name for this feature.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyFeature;
    impl Feature for DummyFeature {
        fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
            vec![0.0; data.len()]
        }
        fn name(&self) -> &str {
            "dummy"
        }
    }

    #[test]
    fn test_feature_trait_can_be_implemented() {
        let f = DummyFeature;
        let candles = vec![OHLCVCandle::new(0, 1.0, 2.0, 0.5, 1.5, 100.0)];
        let result = f.compute(&candles);
        assert_eq!(result.len(), 1);
        assert_eq!(f.name(), "dummy");
    }
}
