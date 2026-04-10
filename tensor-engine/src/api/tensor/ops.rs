use crate::api::error::TensorResult;

/// Tensor computation interface.
pub trait TensorOps {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> crate::api::dtype::DType;
    fn matmul(&self, other: &Self) -> TensorResult<Self> where Self: Sized;
    fn add(&self, other: &Self) -> TensorResult<Self> where Self: Sized;
    fn softmax(&self, dim: i64) -> TensorResult<Self> where Self: Sized;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::dtype::DType;

    #[test]
    fn test_tensor_ops_trait_is_implemented_by_tensor() {
        let t = crate::core::tensor::Tensor::zeros(vec![2, 3]);
        let ops: &dyn TensorOps = &t;
        assert_eq!(ops.shape(), &[2, 3]);
        assert_eq!(ops.dtype(), DType::F32);
    }
}
