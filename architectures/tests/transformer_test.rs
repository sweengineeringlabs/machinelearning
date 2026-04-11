use swe_ml_architectures::*;

/// @covers: TimeSeriesTransformer::new, TimeSeriesTransformer::forward
#[test]
fn test_transformer_end_to_end_forward() {
    let mut model = TimeSeriesTransformer::new(3, 8, 2, 1, 1, 50, 0.0);
    model.eval();
    let input = Tensor::randn([2, 10, 3]);
    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape(), &[2, 1]);
    for &v in &output.to_vec() {
        assert!(v.is_finite(), "Transformer output should be finite, got {}", v);
    }
}

/// @covers: TimeSeriesTransformer::parameters
#[test]
fn test_transformer_has_learnable_parameters() {
    let model = TimeSeriesTransformer::new(3, 8, 2, 1, 1, 50, 0.0);
    let params = model.parameters();
    assert!(!params.is_empty());
    let total: usize = params.iter().map(|p| p.numel()).sum();
    assert!(total > 0);
}
