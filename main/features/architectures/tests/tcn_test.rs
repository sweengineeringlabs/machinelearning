use swe_ml_architectures::*;

/// @covers: TCN::new, TCN::forward
#[test]
fn test_tcn_end_to_end_forward() {
    let mut tcn = TCN::new(3, 1, 8, 3, 2);
    let input = Tensor::randn([1, 3, 20]); // batch=1, features=3, seq_len=20
    let output = tcn.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 1]);
    let val = output.to_vec()[0];
    assert!(val.is_finite(), "TCN output should be finite, got {}", val);
}

/// @covers: TCN::parameters
#[test]
fn test_tcn_has_learnable_parameters() {
    let tcn = TCN::new(2, 4, 8, 3, 2);
    let params = tcn.parameters();
    assert!(!params.is_empty());
    for p in &params {
        assert!(p.requires_grad(), "TCN parameters should require grad");
    }
}
