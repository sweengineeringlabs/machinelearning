// Re-exports of all public trait types from the api layer.

pub use crate::api::layer::Layer;
pub use crate::api::loss::Loss;
pub use crate::api::optim::{LRScheduler, Optimizer};
pub use crate::api::tape::BackwardOp;

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: Layer (trait re-export)
    #[test]
    fn test_layer_trait_is_accessible() {
        fn _assert_layer<T: Layer>() {}
    }

    /// @covers: Loss (trait re-export)
    #[test]
    fn test_loss_trait_is_accessible() {
        fn _assert_loss<T: Loss>() {}
    }

    /// @covers: Optimizer (trait re-export)
    #[test]
    fn test_optimizer_trait_is_accessible() {
        fn _assert_optimizer<T: Optimizer>() {}
    }

    /// @covers: LRScheduler (trait re-export)
    #[test]
    fn test_lr_scheduler_trait_is_accessible() {
        fn _assert_lr_scheduler<T: LRScheduler>() {}
    }

    /// @covers: BackwardOp (trait re-export)
    #[test]
    fn test_backward_op_trait_is_accessible() {
        fn _assert_backward_op<T: BackwardOp>() {}
    }
}
