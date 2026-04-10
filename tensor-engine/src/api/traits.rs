use crate::api::error::TensorResult;

/// Tensor computation interface
pub trait TensorOps {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> crate::api::dtype::DType;
    fn matmul(&self, other: &Self) -> TensorResult<Self> where Self: Sized;
    fn add(&self, other: &Self) -> TensorResult<Self> where Self: Sized;
    fn softmax(&self, dim: i64) -> TensorResult<Self> where Self: Sized;
}
