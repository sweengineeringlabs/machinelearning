use swe_ml_architectures::*;

/// @covers: Layer trait (re-exported via saf)
#[test]
fn test_lstm_implements_layer_trait() {
    // Verify LSTM can be used as a Layer
    let mut lstm = LSTM::new(3, 4, 1);
    let input = Tensor::randn([1, 2, 3]);
    let output = lstm.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 2, 4]);
}

/// @covers: Layer trait (re-exported via saf)
#[test]
fn test_nbeats_implements_layer_trait() {
    let mut nbeats = NBeats::new(10, 5, 1, 1, 8, 1);
    let input = Tensor::randn([1, 10]);
    let output = nbeats.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 5]);
}

/// @covers: Layer::parameters
#[test]
fn test_layer_parameters_returns_tensors() {
    let lstm = LSTM::new(3, 4, 1);
    let params = lstm.parameters();
    assert_eq!(params.len(), 4); // 1 layer * 4 params
    for p in &params {
        assert!(p.requires_grad(), "layer parameters should require grad");
    }
}
