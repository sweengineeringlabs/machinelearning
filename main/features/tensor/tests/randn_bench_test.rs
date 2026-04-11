//! Benchmark randn performance.

use swe_ml_tensor::Tensor;
use std::time::Instant;

#[test]
fn test_randn_200x200_throughput() {
    // Warmup
    for _ in 0..10 {
        let _ = Tensor::randn(vec![200, 200]);
    }

    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = Tensor::randn(vec![200, 200]);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / iterations;

    eprintln!("randn 200x200: {:?} per call ({} iterations, total {:?})", per_call, iterations, elapsed);
    eprintln!("randn 200x200: {:.1} us", per_call.as_nanos() as f64 / 1000.0);
}

#[test]
fn test_randn_1000x1000_throughput() {
    // Warmup
    for _ in 0..3 {
        let _ = Tensor::randn(vec![1000, 1000]);
    }

    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = Tensor::randn(vec![1000, 1000]);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / iterations;

    eprintln!("randn 1000x1000: {:?} per call ({} iterations, total {:?})", per_call, iterations, elapsed);
    eprintln!("randn 1000x1000: {:.1} us", per_call.as_nanos() as f64 / 1000.0);
}

#[test]
fn test_randn_correctness_mean_and_std() {
    let t = Tensor::randn(vec![100000]);
    let data = t.as_slice_f32().unwrap();

    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();

    eprintln!("randn 100K samples: mean={:.4}, std={:.4}", mean, std_dev);

    // Standard normal: mean ≈ 0, std ≈ 1
    assert!(mean.abs() < 0.02, "Mean {} too far from 0", mean);
    assert!((std_dev - 1.0).abs() < 0.02, "Std {} too far from 1.0", std_dev);
}
