use crate::api::feature::Feature;
use crate::api::candle::OHLCVCandle;

/// Simple Moving Average (SMA) of close prices over a rolling window.
///
/// Elements before the window is full are filled with the partial average.
pub struct MovingAverage {
    pub window: usize,
}

impl MovingAverage {
    pub fn new(window: usize) -> Self {
        Self { window }
    }
}

impl Feature for MovingAverage {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        if data.is_empty() || self.window == 0 {
            return vec![0.0; data.len()];
        }
        let mut out = Vec::with_capacity(data.len());
        let mut sum: f32 = 0.0;
        for (i, candle) in data.iter().enumerate() {
            sum += candle.close;
            if i >= self.window {
                sum -= data[i - self.window].close;
                out.push(sum / self.window as f32);
            } else {
                // Partial window: average of available elements
                out.push(sum / (i + 1) as f32);
            }
        }
        out
    }

    fn name(&self) -> &str {
        "moving_average"
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
    fn test_compute_moving_average_basic() {
        let candles = make_candles_with_prices(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let ma = MovingAverage::new(3);
        let result = ma.compute(&candles);
        assert_eq!(result.len(), 5);

        // Partial windows
        assert!((result[0] - 10.0).abs() < 1e-6); // avg(10)
        assert!((result[1] - 15.0).abs() < 1e-6); // avg(10, 20)

        // Full window
        assert!((result[2] - 20.0).abs() < 1e-6); // avg(10, 20, 30)
        assert!((result[3] - 30.0).abs() < 1e-6); // avg(20, 30, 40)
        assert!((result[4] - 40.0).abs() < 1e-6); // avg(30, 40, 50)
    }

    /// @covers: name
    #[test]
    fn test_name_returns_moving_average() {
        assert_eq!(MovingAverage::new(5).name(), "moving_average");
    }
}
