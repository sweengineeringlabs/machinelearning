//! Tests for the optim module.
use training_engine::*;

/// @covers: Optimizer + LRScheduler re-exports
#[test]
fn test_optim_reexports_accessible() {
    let sgd = SGD::new(0.01);
    assert!((sgd.lr() - 0.01).abs() < f32::EPSILON);
}
