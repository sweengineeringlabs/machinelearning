//! Admission control interface.
//!
//! A `Throttle` limits how many requests can hold compute resources at once.
//! Implementations decide the backing mechanism (semaphore, token bucket, etc.).

/// A permit held for the duration of one request's compute work.
/// The throttle slot is released when the permit is dropped.
pub struct Permit {
    /// Implementation-held resource whose `Drop` releases the slot.
    _slot: Box<dyn Send + Sync>,
}

impl Permit {
    pub(crate) fn new(slot: Box<dyn Send + Sync>) -> Self {
        Self { _slot: slot }
    }
}

/// Admission control for concurrent request handling.
pub trait Throttle: Send + Sync {
    /// Try to acquire a permit without waiting. Returns `None` if at capacity.
    fn try_acquire(&self) -> Option<Permit>;

    /// Maximum concurrent permits this throttle allows.
    fn capacity(&self) -> usize;

    /// Permits currently available (capacity minus in-flight).
    fn available(&self) -> usize;
}
