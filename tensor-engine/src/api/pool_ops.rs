/// Memory pool interface for reusable byte buffers.
pub trait PoolOps {
    /// Get a buffer of at least `size` bytes.
    fn get(&mut self, size: usize) -> Vec<u8>;
    /// Return a buffer to the pool for future reuse.
    fn put(&mut self, buf: Vec<u8>);
    /// Returns the number of buffers currently in the pool.
    fn len(&self) -> usize;
    /// Returns true if the pool has no cached buffers.
    fn is_empty(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_ops_trait_is_implemented_by_tensor_pool() {
        let mut pool = crate::core::arena::tensor_pool::TensorPool::new(4);
        let ops: &mut dyn PoolOps = &mut pool;
        assert!(ops.is_empty());
        assert_eq!(ops.len(), 0);
        let buf = ops.get(64);
        assert_eq!(buf.len(), 64);
        ops.put(buf);
        assert_eq!(ops.len(), 1);
    }
}
