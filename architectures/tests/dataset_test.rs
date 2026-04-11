use swe_ml_architectures::*;

/// @covers: TimeSeriesDataset::new, TimeSeriesDataset::len
#[test]
fn test_dataset_construction_and_length() {
    let candles: Vec<OHLCVCandle> = (0..20)
        .map(|i| {
            let v = i as f32;
            OHLCVCandle::new(i as i64, v, v + 1.0, v - 0.5, v + 0.5, (i * 100) as f32)
        })
        .collect();

    let ds = TimeSeriesDataset::new(candles, 5, 1);
    assert_eq!(ds.len(), 15); // 20 - (5 + 1 - 1) = 15
    assert!(!ds.is_empty());
}

/// @covers: TimeSeriesDataset::get
#[test]
fn test_dataset_get_returns_correct_shapes() {
    let candles: Vec<OHLCVCandle> = (0..10)
        .map(|i| {
            let v = i as f32;
            OHLCVCandle::new(i as i64, v, v + 1.0, v - 0.5, v + 0.5, (i * 100) as f32)
        })
        .collect();

    let ds = TimeSeriesDataset::new(candles, 3, 1);
    let (input, target) = ds.get(0).unwrap();
    assert_eq!(input.shape(), &[3, 5]); // window=3, features=5
    assert_eq!(target.shape(), &[1]);   // Close target
}

/// @covers: TimeSeriesDataset::split
#[test]
fn test_dataset_split_preserves_total_data() {
    let candles: Vec<OHLCVCandle> = (0..100)
        .map(|i| {
            let v = i as f32;
            OHLCVCandle::new(i as i64, v, v + 1.0, v - 0.5, v + 0.5, (i * 100) as f32)
        })
        .collect();

    let ds = TimeSeriesDataset::new(candles, 5, 1);
    let (train, val) = ds.split(0.8);
    assert_eq!(train.data().len() + val.data().len(), 100);
}
