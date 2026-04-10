// SEA interface contract — central trait re-exports.
// Individual trait definitions live in their own files per rule 108.

pub use crate::api::tensor::ops::TensorOps;
pub use crate::api::pool_ops::PoolOps;
pub use crate::api::config_ops::ConfigOps;
pub use crate::api::quant::ops::QuantOps;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_ops_reexport_accessible() {
        let t = crate::core::tensor::Tensor::zeros(vec![2, 3]);
        let ops: &dyn TensorOps = &t;
        assert_eq!(ops.shape(), &[2, 3]);
    }

    #[test]
    fn test_pool_ops_reexport_accessible() {
        let mut pool = crate::core::arena::tensor_pool::TensorPool::new(4);
        let ops: &mut dyn PoolOps = &mut pool;
        assert!(ops.is_empty());
    }
}
