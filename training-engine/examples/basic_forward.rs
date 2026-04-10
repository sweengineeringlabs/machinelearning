//! Basic forward pass through a linear layer.

use training_engine::{Linear, Layer, Tensor};

fn main() {
    let mut layer = Linear::new(3, 2);
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let output = layer.forward(&input).unwrap();
    println!("Input shape:  {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());
}
