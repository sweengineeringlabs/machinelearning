use tensor_engine::{Tensor, DType};

fn main() {
    // Create tensors
    let a = Tensor::ones([2, 3]);
    let b = Tensor::randn([2, 3]);

    println!("a shape: {:?}, dtype: {:?}", a.shape(), a.dtype());
    println!("b shape: {:?}, dtype: {:?}", b.shape(), b.dtype());

    // Element-wise add
    let c = a.add(&b).expect("add failed");
    println!("a + b shape: {:?}", c.shape());

    // Matmul
    let x = Tensor::randn([2, 3]);
    let w = Tensor::randn([3, 4]);
    let y = x.matmul(&w).expect("matmul failed");
    println!("matmul [2,3] x [3,4] = {:?}", y.shape());

    // F16 conversion
    let f16 = x.to_f16().expect("to_f16 failed");
    println!("F16 dtype: {:?}", f16.dtype());
}
