//! Basic LSTM forward pass example.

use swe_ml_architectures::{LSTM, Layer, Tensor};

fn main() {
    let mut lstm = LSTM::new(4, 8, 1);
    // batch=1, seq_len=3, input_size=4
    let input = Tensor::from_vec(vec![0.1; 12], vec![1, 3, 4]).unwrap();
    let output = lstm.forward(&input).unwrap();
    println!("Input shape:  {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());
}
