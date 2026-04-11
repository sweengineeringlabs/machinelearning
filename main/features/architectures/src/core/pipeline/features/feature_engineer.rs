use crate::api::feature::Feature;
use crate::api::candle::OHLCVCandle;

/// Manages a collection of features and batch-computes them.
pub struct FeatureEngineer {
    features: Vec<Box<dyn Feature>>,
}

impl FeatureEngineer {
    /// Create an empty feature engineer.
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Add a feature. Builder-style.
    pub fn add(mut self, feature: Box<dyn Feature>) -> Self {
        self.features.push(feature);
        self
    }

    /// Compute all features over the given candle data.
    ///
    /// Returns one `Vec<f32>` per feature, each of length `data.len()`.
    pub fn compute_all(&self, data: &[OHLCVCandle]) -> Vec<Vec<f32>> {
        self.features.iter().map(|f| f.compute(data)).collect()
    }

    /// Number of registered features.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether any features are registered.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pipeline::features::returns::Returns;
    use crate::core::pipeline::features::moving_average::MovingAverage;
    use crate::core::pipeline::features::volatility::Volatility;
    use crate::core::pipeline::features::rsi::RSI;

    fn make_candles_with_prices(prices: &[f32]) -> Vec<OHLCVCandle> {
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| OHLCVCandle::new(i as i64, p, p + 1.0, p - 1.0, p, 1000.0))
            .collect()
    }

    /// @covers: compute_all
    #[test]
    fn test_compute_all_returns_correct_count() {
        let prices: Vec<f32> = (0..30).map(|i| 100.0 + i as f32 * 0.5).collect();
        let candles = make_candles_with_prices(&prices);

        let engine = FeatureEngineer::new()
            .add(Box::new(Returns))
            .add(Box::new(MovingAverage::new(5)))
            .add(Box::new(Volatility::new(5)))
            .add(Box::new(RSI::new(14)));

        assert_eq!(engine.len(), 4);
        assert!(!engine.is_empty());

        let results = engine.compute_all(&candles);
        assert_eq!(results.len(), 4);
        for result in &results {
            assert_eq!(result.len(), candles.len());
        }
    }

    /// @covers: new
    #[test]
    fn test_new_creates_empty_engineer() {
        let engine = FeatureEngineer::new();
        assert!(engine.is_empty());
        assert_eq!(engine.len(), 0);
        let results = engine.compute_all(&[]);
        assert!(results.is_empty());
    }

    /// @covers: add
    #[test]
    fn test_add_increases_feature_count() {
        let engine = FeatureEngineer::new()
            .add(Box::new(Returns));
        assert_eq!(engine.len(), 1);
        let engine = engine.add(Box::new(MovingAverage::new(5)));
        assert_eq!(engine.len(), 2);
    }

    /// @covers: len
    #[test]
    fn test_len_matches_added_features() {
        let engine = FeatureEngineer::new()
            .add(Box::new(Returns))
            .add(Box::new(MovingAverage::new(3)))
            .add(Box::new(Volatility::new(5)));
        assert_eq!(engine.len(), 3);
    }

    /// @covers: is_empty
    #[test]
    fn test_is_empty_false_after_add() {
        let engine = FeatureEngineer::new().add(Box::new(Returns));
        assert!(!engine.is_empty());
    }
}
