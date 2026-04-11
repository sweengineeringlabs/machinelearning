use crate::api::feature::Feature;
use crate::api::candle::OHLCVCandle;
use super::returns::Returns;

/// Rolling standard deviation of log returns (volatility).
///
/// Computed over a rolling window of log returns. Elements before the window
/// is full use a partial window. The very first element is `0.0`.
pub struct Volatility {
    pub window: usize,
}

impl Volatility {
    pub fn new(window: usize) -> Self {
        Self { window }
    }
}

impl Feature for Volatility {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        if data.is_empty() || self.window == 0 {
            return vec![0.0; data.len()];
        }

        // First compute log returns
        let returns = Returns.compute(data);

        let mut out = Vec::with_capacity(data.len());
        out.push(0.0); // First element has no return

        for i in 1..data.len() {
            let start = if i >= self.window { i - self.window + 1 } else { 1 };
            let window_returns = &returns[start..=i];
            let n = window_returns.len() as f32;
            if n <= 1.0 {
                out.push(0.0);
                continue;
            }
            let mean = window_returns.iter().sum::<f32>() / n;
            let variance =
                window_returns.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / (n - 1.0);
            out.push(variance.sqrt());
        }
        out
    }

    fn name(&self) -> &str {
        "volatility"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles_with_prices(prices: &[f32]) -> Vec<OHLCVCandle> {
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| OHLCVCandle::new(i as i64, p, p + 1.0, p - 1.0, p, 1000.0))
            .collect()
    }

    /// @covers: compute
    #[test]
    fn test_compute_constant_prices_zero_volatility() {
        let candles = make_candles_with_prices(&[100.0, 100.0, 100.0, 100.0, 100.0]);
        let vol = Volatility::new(3);
        let result = vol.compute(&candles);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-6);
        for v in &result {
            assert!(*v >= 0.0);
            assert!(*v < 1e-6);
        }
    }

    /// @covers: compute
    #[test]
    fn test_compute_varying_prices_positive_volatility() {
        let candles = make_candles_with_prices(&[100.0, 110.0, 95.0, 108.0, 102.0]);
        let vol = Volatility::new(3);
        let result = vol.compute(&candles);
        assert!(result[2] > 0.0);
        assert!(result[3] > 0.0);
        assert!(result[4] > 0.0);
    }

    /// @covers: name
    #[test]
    fn test_name_returns_volatility() {
        assert_eq!(Volatility::new(5).name(), "volatility");
    }
}
