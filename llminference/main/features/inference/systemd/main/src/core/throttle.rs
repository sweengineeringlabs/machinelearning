//! Semaphore-backed `Throttle` implementation.

use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::api::throttle::{Permit, Throttle};

/// Limits concurrency via a `tokio::sync::Semaphore`.
pub struct SemaphoreThrottle {
    semaphore: Arc<Semaphore>,
    capacity: usize,
}

impl SemaphoreThrottle {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            semaphore: Arc::new(Semaphore::new(capacity)),
            capacity,
        }
    }
}

impl Throttle for SemaphoreThrottle {
    fn try_acquire(&self) -> Option<Permit> {
        // Owned permit — released when dropped from inside the Box.
        let owned = Arc::clone(&self.semaphore).try_acquire_owned().ok()?;
        Some(Permit::new(Box::new(owned)))
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn available(&self) -> usize {
        self.semaphore.available_permits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_acquire_admits_up_to_capacity_then_rejects() {
        let throttle = SemaphoreThrottle::new(2);
        let p1 = throttle.try_acquire();
        let p2 = throttle.try_acquire();
        let p3 = throttle.try_acquire();
        assert!(p1.is_some());
        assert!(p2.is_some());
        assert!(p3.is_none(), "third acquire must fail at capacity=2");
        assert_eq!(throttle.capacity(), 2);
        assert_eq!(throttle.available(), 0);
    }

    #[test]
    fn test_dropping_permit_releases_slot() {
        let throttle = SemaphoreThrottle::new(1);
        let p1 = throttle.try_acquire();
        assert!(p1.is_some());
        assert_eq!(throttle.available(), 0);
        drop(p1);
        assert_eq!(throttle.available(), 1);
        assert!(throttle.try_acquire().is_some(), "slot must be reusable after drop");
    }

    #[test]
    fn test_new_clamps_zero_to_one() {
        let throttle = SemaphoreThrottle::new(0);
        assert_eq!(throttle.capacity(), 1);
    }
}
