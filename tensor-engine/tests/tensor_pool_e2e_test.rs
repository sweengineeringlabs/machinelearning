//! E2E tests for TensorPool.

use tensor_engine::TensorPool;

/// @covers: TensorPool::new
#[test]
fn test_tensor_pool_new_is_empty() {
    let pool = TensorPool::new(8);
    assert!(pool.is_empty());
    assert_eq!(pool.len(), 0);
}

/// @covers: TensorPool::get
#[test]
fn test_tensor_pool_get_returns_zeroed_buffer() {
    let mut pool = TensorPool::new(4);
    let buf = pool.get(256);
    assert_eq!(buf.len(), 256);
    assert!(buf.iter().all(|&b| b == 0));
}

/// @covers: TensorPool::put
#[test]
fn test_tensor_pool_put_and_reuse() {
    let mut pool = TensorPool::new(4);
    let buf = pool.get(1024);
    pool.put(buf);
    assert_eq!(pool.len(), 1);
    let buf2 = pool.get(512);
    assert!(buf2.capacity() >= 1024);
}

/// @covers: TensorPool::len
#[test]
fn test_tensor_pool_len_after_operations() {
    let mut pool = TensorPool::new(4);
    pool.put(vec![0u8; 100]);
    pool.put(vec![0u8; 200]);
    assert_eq!(pool.len(), 2);
}

/// @covers: TensorPool::is_empty
#[test]
fn test_tensor_pool_is_empty_after_drain() {
    let mut pool = TensorPool::new(4);
    pool.put(vec![0u8; 100]);
    let _ = pool.get(50);
    assert!(pool.is_empty());
}
