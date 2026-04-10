/// Memory pooling for tensor allocations.
///
/// TensorPool reuses byte buffers across forward passes to reduce allocation
/// overhead. Buffers returned via `put()` are kept in a pool and returned by
/// `get()` when a matching or larger buffer is available.

use crate::api::traits::PoolOps;

/// A pool of reusable byte buffers for tensor storage.
pub(crate) struct TensorPool {
    buffers: Vec<Vec<u8>>,
    capacity: usize,
}

impl TensorPool {
    /// Create a new pool with the given maximum number of cached buffers.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            buffers: Vec::new(),
            capacity,
        }
    }
}

impl PoolOps for TensorPool {
    fn get(&mut self, size: usize) -> Vec<u8> {
        // Find the smallest buffer that fits
        let mut best_idx = None;
        let mut best_size = usize::MAX;

        for (i, buf) in self.buffers.iter().enumerate() {
            let cap = buf.capacity();
            if cap >= size && cap < best_size {
                best_idx = Some(i);
                best_size = cap;
            }
        }

        if let Some(idx) = best_idx {
            let mut buf = self.buffers.swap_remove(idx);
            buf.clear();
            buf.resize(size, 0);
            buf
        } else {
            vec![0u8; size]
        }
    }

    fn put(&mut self, buf: Vec<u8>) {
        if self.buffers.len() < self.capacity {
            self.buffers.push(buf);
        }
        // else: drop the buffer
    }

    fn len(&self) -> usize {
        self.buffers.len()
    }

    fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: new
    #[test]
    fn test_pool_new() {
        let pool = TensorPool::new(4);
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
    }

    /// @covers: get
    #[test]
    fn test_get_allocates_new_buffer() {
        let mut pool = TensorPool::new(4);
        let buf = pool.get(100);
        assert_eq!(buf.len(), 100);
    }

    /// @covers: put
    #[test]
    fn test_put_stores_buffer_for_reuse() {
        let mut pool = TensorPool::new(4);
        let buf = pool.get(100);
        pool.put(buf);
        assert_eq!(pool.len(), 1);

        // Should reuse the 100-byte buffer
        let buf2 = pool.get(50);
        assert!(buf2.capacity() >= 100);
        assert_eq!(buf2.len(), 50);
        assert_eq!(pool.len(), 0);
    }

    /// @covers: put
    #[test]
    fn test_pool_capacity_limit() {
        let mut pool = TensorPool::new(2);
        pool.put(vec![0u8; 100]);
        pool.put(vec![0u8; 200]);
        pool.put(vec![0u8; 300]); // should be dropped
        assert_eq!(pool.len(), 2);
    }

    /// @covers: is_empty
    #[test]
    fn test_is_empty_true_when_no_buffers() {
        let pool = TensorPool::new(4);
        assert!(pool.is_empty());
    }

    /// @covers: is_empty
    #[test]
    fn test_is_empty_false_after_put() {
        let mut pool = TensorPool::new(4);
        pool.put(vec![0u8; 64]);
        assert!(!pool.is_empty());
    }

    /// @covers: len
    #[test]
    fn test_len_reflects_buffered_count() {
        let mut pool = TensorPool::new(4);
        assert_eq!(pool.len(), 0);
        pool.put(vec![0u8; 100]);
        assert_eq!(pool.len(), 1);
        pool.put(vec![0u8; 200]);
        assert_eq!(pool.len(), 2);
        let _ = pool.get(50);
        assert_eq!(pool.len(), 1);
    }

    /// @covers: get
    #[test]
    fn test_pool_best_fit() {
        let mut pool = TensorPool::new(4);
        pool.put(vec![0u8; 100]);
        pool.put(vec![0u8; 500]);
        pool.put(vec![0u8; 200]);

        // Should pick the 200-byte buffer (smallest that fits 150)
        let buf = pool.get(150);
        assert!(buf.capacity() >= 200);
        assert_eq!(pool.len(), 2);
    }
}
