use crate::api::feature::Feature;
use crate::api::candle::OHLCVCandle;

/// Log returns: `ln(close[i] / close[i-1])`.
///
/// The first element is `0.0` because there is no preceding candle.
pub struct Returns;

impl Feature for Returns {
    fn compute(&self, data: &[OHLCVCandle]) -> Vec<f32> {
        if data.is_empty() {
            return vec![];
        }
        let mut out = Vec::with_capacity(data.len());
        out.push(0.0);
        for i in 1..data.len() {
            let prev = data[i - 1].close;
            let curr = data[i].close;
            if prev > 0.0 {
                out.push((curr / prev).ln());
            } else {
                out.push(0.0);
            }
        }
        out
    }

    fn name(&self) -> &str {
        "returns"
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
    fn test_compute_log_returns_basic() {
        let candles = make_candles_with_prices(&[100.0, 110.0, 105.0]);
        let ret = Returns.compute(&candles);
        assert_eq!(ret.len(), 3);
        assert!((ret[0] - 0.0).abs() < 1e-6);
        assert!((ret[1] - (110.0_f32 / 100.0).ln()).abs() < 1e-5);
        assert!((ret[2] - (105.0_f32 / 110.0).ln()).abs() < 1e-5);
    }

    /// @covers: compute
    #[test]
    fn test_compute_empty_returns_empty() {
        let ret = Returns.compute(&[]);
        assert!(ret.is_empty());
    }

    /// @covers: name
    #[test]
    fn test_name_returns_returns() {
        assert_eq!(Returns.name(), "returns");
    }
}
