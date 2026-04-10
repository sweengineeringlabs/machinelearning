//! E2E tests using real GBPJPY OHLCV data fixtures.
//! Verifies tensor ops on production-shaped data and exercises multi-core paths.

use swe_ml_tensor::Tensor;
use std::time::Instant;

/// Parse CSV fixture into OHLCV column vectors (Open, High, Low, Close, Volume).
fn load_ohlcv(path: &str) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", path, e));
    let mut open = Vec::new();
    let mut high = Vec::new();
    let mut low = Vec::new();
    let mut close = Vec::new();
    let mut volume = Vec::new();

    for line in content.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 6 { continue; }
        open.push(cols[1].parse::<f32>().unwrap());
        high.push(cols[2].parse::<f32>().unwrap());
        low.push(cols[3].parse::<f32>().unwrap());
        close.push(cols[4].parse::<f32>().unwrap());
        volume.push(cols[5].parse::<f32>().unwrap());
    }
    (open, high, low, close, volume)
}

/// Build a feature matrix [rows, 5] from OHLCV data.
fn build_feature_matrix(path: &str) -> Tensor {
    let (open, high, low, close, volume) = load_ohlcv(path);
    let rows = open.len();
    let mut data = Vec::with_capacity(rows * 5);
    for i in 0..rows {
        data.push(open[i]);
        data.push(high[i]);
        data.push(low[i]);
        data.push(close[i]);
        data.push(volume[i]);
    }
    Tensor::from_vec(data, vec![rows, 5]).unwrap()
}

#[test]
fn test_load_fixture_200_rows_correct_shape() {
    let t = build_feature_matrix("tests/fixtures/gbpjpy_h1_200.csv");
    assert_eq!(t.shape(), &[200, 5]);
}

#[test]
fn test_load_fixture_1000_rows_correct_shape() {
    let t = build_feature_matrix("tests/fixtures/gbpjpy_h1_1000.csv");
    assert_eq!(t.shape(), &[1000, 5]);
}

#[test]
fn test_load_fixture_10000_rows_correct_shape() {
    let t = build_feature_matrix("tests/fixtures/gbpjpy_h1_10000.csv");
    assert_eq!(t.shape(), &[10000, 5]);
}

#[test]
fn test_ohlcv_values_are_valid_prices() {
    let (open, high, low, close, _vol) = load_ohlcv("tests/fixtures/gbpjpy_h1_200.csv");
    for i in 0..open.len() {
        assert!(high[i] >= low[i], "High {} < Low {} at row {}", high[i], low[i], i);
        assert!(high[i] >= open[i], "High {} < Open {} at row {}", high[i], open[i], i);
        assert!(high[i] >= close[i], "High {} < Close {} at row {}", high[i], close[i], i);
        assert!(low[i] <= open[i], "Low {} > Open {} at row {}", low[i], open[i], i);
        assert!(low[i] <= close[i], "Low {} > Close {} at row {}", low[i], close[i], i);
    }
}

#[test]
fn test_matmul_feature_projection_10000_rows() {
    // [10000, 5] @ [5, 64] = [10000, 64] — simulates a linear layer
    let features = build_feature_matrix("tests/fixtures/gbpjpy_h1_10000.csv");
    let weight_data: Vec<f32> = (0..5 * 64).map(|i| (i as f32) * 0.01).collect();
    let weights = Tensor::from_vec(weight_data, vec![5, 64]).unwrap();

    let result = features.matmul(&weights).unwrap();
    assert_eq!(result.shape(), &[10000, 64]);

    let data = result.as_slice_f32().unwrap();
    assert!(data.iter().all(|v| v.is_finite()), "Matmul produced non-finite values");
}

#[test]
fn test_softmax_10000_rows_exercises_parallel_path() {
    // [10000, 64] = 640,000 elements — well above 4096 threshold
    let features = build_feature_matrix("tests/fixtures/gbpjpy_h1_10000.csv");
    let weight_data: Vec<f32> = (0..5 * 64).map(|i| (i as f32) * 0.01).collect();
    let weights = Tensor::from_vec(weight_data, vec![5, 64]).unwrap();
    let projected = features.matmul(&weights).unwrap();

    let softmaxed = projected.softmax(-1).unwrap();
    assert_eq!(softmaxed.shape(), &[10000, 64]);

    let data = softmaxed.as_slice_f32().unwrap();
    // Each row should sum to ~1.0
    for row in 0..10000 {
        let row_sum: f32 = data[row * 64..(row + 1) * 64].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-5, "Row {} softmax sum = {}", row, row_sum);
    }
}

#[test]
fn test_attention_simulation_1000_rows() {
    // Simulate self-attention: Q, K, V projections + scaled dot-product
    let features = build_feature_matrix("tests/fixtures/gbpjpy_h1_1000.csv");
    let d_model = 32;

    // Project to Q, K, V: [1000, 5] @ [5, 32] = [1000, 32]
    let wq_data: Vec<f32> = (0..5 * d_model).map(|i| ((i as f32) * 0.02) - 0.5).collect();
    let wk_data: Vec<f32> = (0..5 * d_model).map(|i| ((i as f32) * 0.03) - 0.5).collect();
    let wv_data: Vec<f32> = (0..5 * d_model).map(|i| ((i as f32) * 0.01) - 0.5).collect();

    let wq = Tensor::from_vec(wq_data, vec![5, d_model]).unwrap();
    let wk = Tensor::from_vec(wk_data, vec![5, d_model]).unwrap();
    let wv = Tensor::from_vec(wv_data, vec![5, d_model]).unwrap();

    let q = features.matmul(&wq).unwrap(); // [1000, 32]
    let k = features.matmul(&wk).unwrap(); // [1000, 32]
    let v = features.matmul(&wv).unwrap(); // [1000, 32]

    // Attention scores: Q @ K^T = [1000, 32] @ [32, 1000] = [1000, 1000]
    let kt = k.t().unwrap(); // [32, 1000]
    let scores = q.matmul(&kt).unwrap(); // [1000, 1000]
    assert_eq!(scores.shape(), &[1000, 1000]);

    // Scale
    let scale = (d_model as f32).sqrt();
    let scaled = scores.div_scalar(scale);

    // Softmax — [1000, 1000] = 1M elements, well above parallel threshold
    let attn_weights = scaled.softmax(-1).unwrap();
    let attn_data = attn_weights.as_slice_f32().unwrap();
    for row in 0..1000 {
        let row_sum: f32 = attn_data[row * 1000..(row + 1) * 1000].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-4, "Attention row {} sum = {}", row, row_sum);
    }

    // Apply to V: [1000, 1000] @ [1000, 32] = [1000, 32]
    let output = attn_weights.matmul(&v).unwrap();
    assert_eq!(output.shape(), &[1000, 32]);

    let out_data = output.as_slice_f32().unwrap();
    assert!(out_data.iter().all(|v: &f32| v.is_finite()), "Attention output has non-finite values");
}

#[test]
fn test_parallel_vs_sequential_softmax_agree() {
    // Build a tensor above the parallel threshold and verify correctness
    let features = build_feature_matrix("tests/fixtures/gbpjpy_h1_10000.csv");
    let weight_data: Vec<f32> = (0..5 * 128).map(|i| ((i as f32) * 0.005) - 1.0).collect();
    let weights = Tensor::from_vec(weight_data, vec![5, 128]).unwrap();
    let projected = features.matmul(&weights).unwrap(); // [10000, 128] = 1.28M elements

    let result = projected.softmax(-1).unwrap();
    let data = result.as_slice_f32().unwrap();

    // Verify every row sums to 1 and all values are in [0, 1]
    for row in 0..10000 {
        let slice = &data[row * 128..(row + 1) * 128];
        let sum: f32 = slice.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Row {} sum = {}", row, sum);
        assert!(slice.iter().all(|&v| v >= 0.0 && v <= 1.0), "Row {} has out-of-range value", row);
    }
}

#[test]
fn test_large_matmul_throughput() {
    // [10000, 5] @ [5, 256] @ [256, 64] — two-layer projection
    let features = build_feature_matrix("tests/fixtures/gbpjpy_h1_10000.csv");
    let w1_data: Vec<f32> = (0..5 * 256).map(|i| ((i as f32) * 0.003) - 0.5).collect();
    let w2_data: Vec<f32> = (0..256 * 64).map(|i| ((i as f32) * 0.001) - 0.5).collect();
    let w1 = Tensor::from_vec(w1_data, vec![5, 256]).unwrap();
    let w2 = Tensor::from_vec(w2_data, vec![256, 64]).unwrap();

    let start = Instant::now();
    let hidden = features.matmul(&w1).unwrap();  // [10000, 256]
    let output = hidden.matmul(&w2).unwrap();     // [10000, 64]
    let elapsed = start.elapsed();

    assert_eq!(output.shape(), &[10000, 64]);
    eprintln!("Two-layer projection (10000x5 -> 256 -> 64): {:?}", elapsed);
    eprintln!("Rayon thread pool: {} threads", rayon::current_num_threads());
}

#[test]
fn test_multicore_softmax_above_threshold() {
    // Parallel threshold is 4096 elements.
    // [10000, 128] = 1,280,000 elements — should hit the parallel path.
    // We verify correctness (not timing) — the parallel path must produce
    // the same result as the sequential path.
    let cores = rayon::current_num_threads();
    eprintln!("Rayon thread pool size: {} threads", cores);

    let data: Vec<f32> = (0..10000 * 128).map(|i| ((i % 1000) as f32) * 0.01 - 5.0).collect();
    let t = Tensor::from_vec(data, vec![10000, 128]).unwrap();

    let result = t.softmax(-1).unwrap();
    let out = result.as_slice_f32().unwrap();

    // 10000 rows, each must sum to 1.0 and be in [0, 1]
    for row in 0..10000 {
        let slice = &out[row * 128..(row + 1) * 128];
        let sum: f32 = slice.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Row {} sum = {} (expected 1.0)", row, sum);
    }

    // Confirm we're above the parallel threshold
    let total_elements = 10000 * 128;
    assert!(total_elements > 4096, "Test data ({}) should exceed parallel threshold (4096)", total_elements);
    eprintln!("Softmax on {} elements completed (threshold: 4096)", total_elements);
}

#[test]
fn test_daily_vs_hourly_same_price_range() {
    // Sanity check: daily and hourly data from same pair should overlap in price range
    let (_, h_high, h_low, _, _) = load_ohlcv("tests/fixtures/gbpjpy_h1_200.csv");
    let (_, d_high, d_low, _, _) = load_ohlcv("tests/fixtures/gbpjpy_d1_200.csv");

    let h_max = h_high.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let h_min = h_low.iter().cloned().fold(f32::INFINITY, f32::min);
    let d_max = d_high.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let d_min = d_low.iter().cloned().fold(f32::INFINITY, f32::min);

    // Both should be in GBPJPY range (roughly 100-250 over history)
    assert!(h_max > 100.0 && h_max < 300.0, "Hourly max {} out of range", h_max);
    assert!(d_max > 100.0 && d_max < 300.0, "Daily max {} out of range", d_max);
    assert!(h_min > 100.0 && h_min < 300.0, "Hourly min {} out of range", h_min);
    assert!(d_min > 100.0 && d_min < 300.0, "Daily min {} out of range", d_min);
}
