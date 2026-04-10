use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub(crate) u64);

impl TensorId {
    pub(crate) fn next() -> Self {
        TensorId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorId({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: TensorId::next
    #[test]
    fn test_tensor_id_next_increments() {
        let id1 = TensorId::next();
        let id2 = TensorId::next();
        assert_ne!(id1, id2);
        assert!(id2.0 > id1.0);
    }

    /// @covers: TensorId::Display
    #[test]
    fn test_tensor_id_display_format() {
        let id = TensorId(42);
        assert_eq!(format!("{}", id), "TensorId(42)");
    }
}
