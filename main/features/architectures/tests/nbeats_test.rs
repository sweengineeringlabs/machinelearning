use swe_ml_architectures::*;

/// @covers: NBeats::new
#[test]
fn test_create_nbeats_model() {
    let model = NBeats::new(10, 5, 2, 3, 16, 2);
    assert_eq!(model.backcast_size(), 10);
    assert_eq!(model.forecast_size(), 5);
}

/// @covers: NBeats::forward
#[test]
fn test_nbeats_forward_produces_finite_output() {
    let mut model = NBeats::new(8, 3, 1, 1, 8, 1);
    let input = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![1, 8],
    )
    .unwrap();
    let output = model.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 3]);
    for &v in &output.to_vec() {
        assert!(v.is_finite(), "NBeats output should be finite, got {}", v);
    }
}
