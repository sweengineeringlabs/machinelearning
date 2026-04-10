//! Integration tests for TensorPool (PoolOps trait).

use tensor_engine::TensorPool;

/// @covers: TensorPool::new
#[test]
fn test_pool_new_is_empty() {
    let pool = TensorPool::new(8);
    assert!(pool.is_empty());
    assert_eq!(pool.len(), 0);
}

/// @covers: TensorPool::get
#[test]
fn test_pool_get_returns_zeroed_buffer() {
    let mut pool = TensorPool::new(4);
    let buf = pool.get(256);
    assert_eq!(buf.len(), 256);
    assert!(buf.iter().all(|&b| b == 0), "fresh buffer should be zeroed");
}

/// @covers: TensorPool::put, TensorPool::get
#[test]
fn test_pool_reuses_returned_buffer() {
    let mut pool = TensorPool::new(4);

    // Allocate and return
    let buf = pool.get(1024);
    assert_eq!(pool.len(), 0);
    pool.put(buf);
    assert_eq!(pool.len(), 1);

    // Get again — should reuse the 1024-capacity buffer
    let buf2 = pool.get(512);
    assert!(buf2.capacity() >= 1024, "reused buffer should have original capacity");
    assert_eq!(buf2.len(), 512);
    assert_eq!(pool.len(), 0);
}

/// @covers: TensorPool::put
#[test]
fn test_pool_drops_excess_beyond_capacity() {
    let mut pool = TensorPool::new(2);
    pool.put(vec![0u8; 100]);
    pool.put(vec![0u8; 200]);
    pool.put(vec![0u8; 300]); // exceeds capacity
    assert_eq!(pool.len(), 2, "pool should cap at configured capacity");
}

/// @covers: TensorPool::get
#[test]
fn test_pool_allocates_when_no_suitable_buffer() {
    let mut pool = TensorPool::new(4);
    pool.put(vec![0u8; 64]); // too small
    let buf = pool.get(128);
    assert_eq!(buf.len(), 128);
    // The 64-byte buffer stays in the pool
    assert_eq!(pool.len(), 1);
}

/// @covers: TensorPool::is_empty
#[test]
fn test_pool_is_empty_after_draining() {
    let mut pool = TensorPool::new(4);
    pool.put(vec![0u8; 100]);
    let _ = pool.get(50); // drains the one buffer
    assert!(pool.is_empty());
}
