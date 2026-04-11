use crate::api::feature::Feature;
use crate::api::candle::OHLCVCandle;

/// Relative Strength Index using Wilder's Exponential Moving Average.
///
/// Implementation:
/// 1. Compute price changes: `delta[i] = close[i] - close[i-1]`.
/// 2. Seed the first average gain and loss from the simple mean of the
///    first `period` deltas.
/// 3. Subsequent values use Wilder's smoothing:
///    `avg_gain = avg_gain * (1 - alpha) + gain * alpha` where `alpha = 1/period`.
/// 4. `RSI = 100 - 100 / (1 + avg_gain / avg_loss)`.
///
/// Elements before `period` candles are available are set to `50.0` (neutral).
pub struct RSI {
    pub period: usize,
}

impl RSI {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Feature for RSI {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        let n = data.len();
        if n == 0 || self.period == 0 {
            return vec![50.0; n];
        }

        let mut out = vec![50.0; n];

        // Need at least period+1 data points (period deltas) to seed
        if n <= self.period {
            return out;
        }

        // Compute price deltas
        let deltas: Vec<f32> = (1..n).map(|i| data[i].close - data[i - 1].close).collect();

        // Seed: SMA of first `period` gains and losses
        let mut avg_gain: f32 = 0.0;
        let mut avg_loss: f32 = 0.0;
        for i in 0..self.period {
            let d = deltas[i];
            if d > 0.0 {
                avg_gain += d;
            } else {
                avg_loss += -d;
            }
        }
        avg_gain /= self.period as f32;
        avg_loss /= self.period as f32;

        // First RSI value at index = period (corresponds to delta index period-1)
        let rsi_val = if avg_loss == 0.0 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        };
        out[self.period] = rsi_val;

        // Wilder's smoothing for subsequent values
        let alpha = 1.0 / self.period as f32;
        for i in self.period..deltas.len() {
            let d = deltas[i];
            let gain = if d > 0.0 { d } else { 0.0 };
            let loss = if d < 0.0 { -d } else { 0.0 };

            avg_gain = avg_gain * (1.0 - alpha) + gain * alpha;
            avg_loss = avg_loss * (1.0 - alpha) + loss * alpha;

            let rsi_val = if avg_loss == 0.0 {
                100.0
            } else {
                100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            };
            // delta index i corresponds to data index i+1
            out[i + 1] = rsi_val;
        }

        out
    }

    fn name(&self) -> &str {
        "rsi"
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
    fn test_compute_all_gains_rsi_near_100() {
        let prices: Vec<f32> = (0..20).map(|i| 100.0 + i as f32).collect();
        let candles = make_candles_with_prices(&prices);
        let rsi = RSI::new(14);
        let result = rsi.compute(&candles);
        assert!((result[14] - 100.0).abs() < 1e-3);
        assert!((result[19] - 100.0).abs() < 1e-3);
    }

    /// @covers: compute
    #[test]
    fn test_compute_all_losses_rsi_near_0() {
        let prices: Vec<f32> = (0..20).map(|i| 200.0 - i as f32).collect();
        let candles = make_candles_with_prices(&prices);
        let rsi = RSI::new(14);
        let result = rsi.compute(&candles);
        assert!((result[14] - 0.0).abs() < 1e-3);
    }

    /// @covers: compute
    #[test]
    fn test_compute_neutral_before_period() {
        let prices: Vec<f32> = (0..20).map(|i| 100.0 + (i as f32).sin() * 5.0).collect();
        let candles = make_candles_with_prices(&prices);
        let rsi = RSI::new(14);
        let result = rsi.compute(&candles);
        for i in 0..14 {
            assert!((result[i] - 50.0).abs() < 1e-6);
        }
    }

    /// @covers: compute
    #[test]
    fn test_compute_values_in_0_to_100_range() {
        let prices: Vec<f32> = (0..50)
            .map(|i| 100.0 + (i as f32 * 0.7).sin() * 20.0)
            .collect();
        let candles = make_candles_with_prices(&prices);
        let rsi = RSI::new(14);
        let result = rsi.compute(&candles);
        for val in &result {
            assert!(*val >= 0.0 && *val <= 100.0, "RSI out of range: {}", val);
        }
    }

    /// @covers: name
    #[test]
    fn test_name_returns_rsi() {
        assert_eq!(RSI::new(14).name(), "rsi");
    }
}
