/// Time series dataset for windowed sequence-to-target learning.
///
/// FR-701: Windowed time series dataset construction.
/// FR-702: Configurable target columns and train/val splitting.
use crate::api::error::{SwetsError, SwetsResult};
use crate::api::tensor::Tensor;
use crate::api::candle::OHLCVCandle;
use super::target_column::TargetColumn;

/// A windowed time series dataset that produces (input, target) pairs.
///
/// Each sample consists of:
/// - Input: a window of `window_size` candles with selected features,
///   shaped `[window_size, num_features]`.
/// - Target: the target column(s) extracted from the candle at offset
///   `idx + window_size + prediction_horizon - 1`.
#[derive(Debug, Clone)]
pub struct TimeSeriesDataset {
    data: Vec<OHLCVCandle>,
    window_size: usize,
    prediction_horizon: usize,
    features: Vec<String>,
    target_columns: TargetColumn,
}

impl TimeSeriesDataset {
    /// Create a new dataset with default features and `Close` target.
    ///
    /// Default features: `["open", "high", "low", "close", "volume"]`.
    pub fn new(data: Vec<OHLCVCandle>, window_size: usize, prediction_horizon: usize) -> Self {
        Self {
            data,
            window_size,
            prediction_horizon,
            features: vec![
                "open".to_string(),
                "high".to_string(),
                "low".to_string(),
                "close".to_string(),
                "volume".to_string(),
            ],
            target_columns: TargetColumn::Close,
        }
    }

    /// Override the target columns.
    pub fn with_targets(mut self, targets: TargetColumn) -> Self {
        self.target_columns = targets;
        self
    }

    /// Override the input features.
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }

    /// Number of valid samples in the dataset.
    ///
    /// A sample requires `window_size` candles for the input window plus
    /// `prediction_horizon` candles for the target lookahead. The last usable
    /// starting index is `data.len() - window_size - prediction_horizon`, giving
    /// `data.len() - window_size - prediction_horizon + 1` total samples.
    pub fn len(&self) -> usize {
        self.data
            .len()
            .saturating_sub(self.window_size + self.prediction_horizon - 1)
    }

    /// Whether the dataset contains no valid samples.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the (input, target) pair for sample at `idx`.
    ///
    /// - Input shape: `[window_size, num_features]`
    /// - Target shape: `[target_dim]` where `target_dim` depends on `TargetColumn`
    pub fn get(&self, idx: usize) -> SwetsResult<(Tensor, Tensor)> {
        if idx >= self.len() {
            return Err(SwetsError::InvalidConfig(format!(
                "Index {} out of bounds for dataset of length {}",
                idx,
                self.len()
            )));
        }

        let num_features = self.features.len();

        // Build input window: candles[idx .. idx + window_size]
        let mut input_data = Vec::with_capacity(self.window_size * num_features);
        for i in idx..idx + self.window_size {
            let candle = &self.data[i];
            for feat_name in &self.features {
                let val = candle.get_feature(feat_name).ok_or_else(|| {
                    SwetsError::InvalidConfig(format!("Unknown feature column: {}", feat_name))
                })?;
                input_data.push(val);
            }
        }

        let input = Tensor::from_vec(input_data, vec![self.window_size, num_features])
            .map_err(SwetsError::TensorError)?;

        // Build target: candle at idx + window_size + prediction_horizon - 1
        let target_idx = idx + self.window_size + self.prediction_horizon - 1;
        let target_candle = &self.data[target_idx];

        let target_data = match &self.target_columns {
            TargetColumn::Close => {
                vec![target_candle.close]
            }
            TargetColumn::Single(col) => {
                let val = target_candle.get_feature(col).ok_or_else(|| {
                    SwetsError::InvalidConfig(format!("Unknown target column: {}", col))
                })?;
                vec![val]
            }
            TargetColumn::Multi(cols) => {
                let mut vals = Vec::with_capacity(cols.len());
                for col in cols {
                    let val = target_candle.get_feature(col).ok_or_else(|| {
                        SwetsError::InvalidConfig(format!("Unknown target column: {}", col))
                    })?;
                    vals.push(val);
                }
                vals
            }
        };

        let target_dim = target_data.len();
        let target =
            Tensor::from_vec(target_data, vec![target_dim]).map_err(SwetsError::TensorError)?;

        Ok((input, target))
    }

    /// Split the dataset into two parts at the given ratio.
    ///
    /// `ratio` is the fraction of the *data* that goes to the first (train) split.
    /// The split is performed on the raw candle data to preserve sequential ordering.
    pub fn split(self, ratio: f32) -> (Self, Self) {
        let split_idx = ((self.data.len() as f32) * ratio).round() as usize;
        let split_idx = split_idx.min(self.data.len());

        let (first_data, second_data) = self.data.split_at(split_idx);

        let first = TimeSeriesDataset {
            data: first_data.to_vec(),
            window_size: self.window_size,
            prediction_horizon: self.prediction_horizon,
            features: self.features.clone(),
            target_columns: self.target_columns.clone(),
        };

        let second = TimeSeriesDataset {
            data: second_data.to_vec(),
            window_size: self.window_size,
            prediction_horizon: self.prediction_horizon,
            features: self.features,
            target_columns: self.target_columns,
        };

        (first, second)
    }

    /// Access the underlying candle data.
    pub fn data(&self) -> &[OHLCVCandle] {
        &self.data
    }

    /// The window size used for input sequences.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// The prediction horizon (lookahead).
    pub fn prediction_horizon(&self) -> usize {
        self.prediction_horizon
    }

    /// The feature column names.
    pub fn features(&self) -> &[String] {
        &self.features
    }

    /// The number of feature columns.
    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    /// The target column specification.
    pub fn target_columns(&self) -> &TargetColumn {
        &self.target_columns
    }

    /// Dimensionality of the target vector.
    pub fn target_dim(&self) -> usize {
        match &self.target_columns {
            TargetColumn::Close => 1,
            TargetColumn::Single(_) => 1,
            TargetColumn::Multi(cols) => cols.len(),
        }
    }
}

impl swe_ml_training::Dataset for TimeSeriesDataset {
    fn len(&self) -> usize {
        self.len()
    }

    fn get(&self, index: usize) -> SwetsResult<(Tensor, Tensor)> {
        self.get(index)
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![self.window_size, self.num_features()]
    }

    fn target_dim(&self) -> usize {
        self.target_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles(n: usize) -> Vec<OHLCVCandle> {
        (0..n)
            .map(|i| {
                let v = i as f32;
                OHLCVCandle::new(i as i64, v, v + 1.0, v - 0.5, v + 0.5, (i * 100) as f32)
            })
            .collect()
    }

    /// @covers: len
    #[test]
    fn test_dataset_len() {
        // 20 candles, window=5, horizon=1 => 20 - (5 + 1 - 1) = 15
        let ds = TimeSeriesDataset::new(make_candles(20), 5, 1);
        assert_eq!(ds.len(), 15);

        // 20 candles, window=5, horizon=3 => 20 - (5 + 3 - 1) = 13
        let ds = TimeSeriesDataset::new(make_candles(20), 5, 3);
        assert_eq!(ds.len(), 13);
    }

    /// @covers: len, is_empty
    #[test]
    fn test_dataset_len_too_small() {
        // 3 candles, window=5, horizon=1 => saturating_sub(5+1-1) = saturating_sub(5) = 0
        let ds = TimeSeriesDataset::new(make_candles(3), 5, 1);
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());
    }

    /// @covers: get
    #[test]
    fn test_dataset_get_default_target() {
        let candles = make_candles(10);
        let ds = TimeSeriesDataset::new(candles, 3, 1);

        // idx=0: input from candles[0..3], target from candle[3]
        let (input, target) = ds.get(0).unwrap();
        assert_eq!(input.shape(), &[3, 5]); // window=3, features=5
        assert_eq!(target.shape(), &[1]); // Close target

        // Target should be candle[3].close = 3.0 + 0.5 = 3.5
        let target_data = target.to_vec();
        assert!((target_data[0] - 3.5).abs() < 1e-6);
    }

    /// @covers: get
    #[test]
    fn test_dataset_get_input_values() {
        let candles = make_candles(10);
        let ds = TimeSeriesDataset::new(candles, 2, 1);

        let (input, _) = ds.get(0).unwrap();
        let data = input.to_vec();
        // candle[0]: open=0, high=1, low=-0.5, close=0.5, volume=0
        // candle[1]: open=1, high=2, low=0.5, close=1.5, volume=100
        assert!((data[0] - 0.0).abs() < 1e-6); // open[0]
        assert!((data[1] - 1.0).abs() < 1e-6); // high[0]
        assert!((data[2] - (-0.5)).abs() < 1e-6); // low[0]
        assert!((data[3] - 0.5).abs() < 1e-6); // close[0]
        assert!((data[4] - 0.0).abs() < 1e-6); // volume[0]
        assert!((data[5] - 1.0).abs() < 1e-6); // open[1]
    }

    /// @covers: get
    #[test]
    fn test_dataset_get_out_of_bounds() {
        let ds = TimeSeriesDataset::new(make_candles(10), 3, 1);
        assert!(ds.get(ds.len()).is_err());
    }

    /// @covers: with_targets
    #[test]
    fn test_with_targets_single() {
        let candles = make_candles(10);
        let ds = TimeSeriesDataset::new(candles, 3, 1)
            .with_targets(TargetColumn::Single("open".to_string()));

        let (_, target) = ds.get(0).unwrap();
        assert_eq!(target.shape(), &[1]);
        // Target candle index=3, open=3.0
        let t = target.to_vec();
        assert!((t[0] - 3.0).abs() < 1e-6);
    }

    /// @covers: with_targets
    #[test]
    fn test_with_targets_multi() {
        let candles = make_candles(10);
        let ds = TimeSeriesDataset::new(candles, 3, 1).with_targets(TargetColumn::Multi(vec![
            "open".to_string(),
            "close".to_string(),
        ]));

        let (_, target) = ds.get(0).unwrap();
        assert_eq!(target.shape(), &[2]);
        let t = target.to_vec();
        assert!((t[0] - 3.0).abs() < 1e-6); // open
        assert!((t[1] - 3.5).abs() < 1e-6); // close
    }

    /// @covers: with_features
    #[test]
    fn test_with_features() {
        let candles = make_candles(10);
        let ds = TimeSeriesDataset::new(candles, 3, 1)
            .with_features(vec!["close".to_string(), "volume".to_string()]);

        let (input, _) = ds.get(0).unwrap();
        assert_eq!(input.shape(), &[3, 2]); // 3 window, 2 features
    }

    /// @covers: split
    #[test]
    fn test_split() {
        let candles = make_candles(100);
        let ds = TimeSeriesDataset::new(candles, 5, 1);
        let (train, val) = ds.split(0.8);

        // 80% of 100 candles = 80 candles for train
        assert_eq!(train.data().len(), 80);
        assert_eq!(val.data().len(), 20);

        // Both should still be functional
        assert!(train.len() > 0);
        assert!(val.len() > 0);
    }

    /// @covers: target_dim
    #[test]
    fn test_target_dim() {
        let ds = TimeSeriesDataset::new(make_candles(10), 3, 1);
        assert_eq!(ds.target_dim(), 1);

        let ds = ds.with_targets(TargetColumn::Multi(vec![
            "open".to_string(),
            "high".to_string(),
            "low".to_string(),
        ]));
        assert_eq!(ds.target_dim(), 3);
    }

    /// @covers: new
    #[test]
    fn test_new_creates_dataset() {
        let ds = TimeSeriesDataset::new(make_candles(10), 3, 1);
        assert_eq!(ds.window_size(), 3);
        assert_eq!(ds.prediction_horizon(), 1);
    }

    /// @covers: window_size
    #[test]
    fn test_window_size() {
        let ds = TimeSeriesDataset::new(make_candles(10), 5, 1);
        assert_eq!(ds.window_size(), 5);
    }

    /// @covers: prediction_horizon
    #[test]
    fn test_prediction_horizon() {
        let ds = TimeSeriesDataset::new(make_candles(10), 3, 2);
        assert_eq!(ds.prediction_horizon(), 2);
    }

    /// @covers: num_features
    #[test]
    fn test_num_features() {
        let ds = TimeSeriesDataset::new(make_candles(10), 3, 1);
        assert_eq!(ds.num_features(), 5); // default 5 OHLCV features
    }

    /// @covers: features
    #[test]
    fn test_features() {
        let ds = TimeSeriesDataset::new(make_candles(10), 3, 1);
        assert_eq!(ds.features().len(), 5);
    }

    /// @covers: target_columns
    #[test]
    fn test_target_columns() {
        let ds = TimeSeriesDataset::new(make_candles(10), 3, 1);
        match ds.target_columns() {
            TargetColumn::Close => {} // expected default
            other => panic!("Expected TargetColumn::Close, got {:?}", other),
        }
    }

    /// @covers: data
    #[test]
    fn test_data() {
        let ds = TimeSeriesDataset::new(make_candles(10), 3, 1);
        assert_eq!(ds.data().len(), 10);
    }

    /// @covers: is_empty
    #[test]
    fn test_is_empty_true_when_insufficient_data() {
        let ds = TimeSeriesDataset::new(make_candles(2), 5, 1);
        assert!(ds.is_empty());
    }

    /// @covers: is_empty
    #[test]
    fn test_is_empty_false_when_sufficient_data() {
        let ds = TimeSeriesDataset::new(make_candles(20), 5, 1);
        assert!(!ds.is_empty());
    }
}
