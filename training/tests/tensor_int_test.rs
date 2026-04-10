//! Integration tests for tensor-engine dependency coverage.

use swe_ml_tensor::Tensor as CoreTensor;
use swe_ml_training::*;

/// @covers: Layer trait + Linear + ReLU integration
#[test]
fn test_linear_relu_forward_produces_non_negative_output() {
    let mut linear = Linear::new(4, 3);
    let mut relu = ReLU::new();

    let input = Tensor::randn([2, 4]);
    let hidden = linear.forward(&input).unwrap();
    let output = relu.forward(&hidden).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    // All outputs should be non-negative after ReLU
    for &v in &output.to_vec() {
        assert!(v >= 0.0, "ReLU output should be non-negative, got {}", v);
    }
}

/// @covers: Loss trait + MSELoss + Tensor integration
#[test]
fn test_mse_loss_forward_e2e() {
    let loss = MSELoss::new();
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let target = Tensor::from_vec(vec![1.5, 2.5, 3.5, 4.5], vec![4]).unwrap();
    let result = loss.forward(&pred, &target).unwrap();
    // MSE = mean(0.25 * 4) = 0.25
    assert!((result.to_vec()[0] - 0.25).abs() < 1e-6);
}

/// @covers: Optimizer trait + SGD + tape integration
#[test]
fn test_sgd_optimizer_e2e() {
    tape::clear_tape();
    let sgd = SGD::new(0.01);
    assert!((sgd.lr() - 0.01).abs() < f32::EPSILON);
}

/// @covers: LRScheduler trait + StepLR + Optimizer integration
#[test]
fn test_step_lr_scheduler_decays_optimizer_lr() {
    let mut sgd = SGD::new(0.1);
    let mut scheduler = StepLR::new(0.1, 5, 0.5);

    for _ in 0..5 {
        scheduler.step(&mut sgd);
    }
    assert!((sgd.lr() - 0.05).abs() < 1e-6);
}

/// @covers: Sequential + parameter aggregation
#[test]
fn test_sequential_aggregates_parameters() {
    let seq = Sequential::new(vec![
        Box::new(Linear::new(4, 3)),
        Box::new(ReLU::new()),
        Box::new(Linear::new(3, 2)),
    ]);
    let params = seq.parameters();
    // Linear(4,3): weight[3,4]=12 + bias[3]=3 = 15
    // ReLU: 0
    // Linear(3,2): weight[2,3]=6 + bias[2]=2 = 8
    // Total params = 4 tensors
    assert_eq!(params.len(), 4);
}

/// @covers: Tensor constructors (tensor-engine dependency)
#[test]
fn test_tensor_constructors_exercise_swe_ml_tensor() {
    let z = Tensor::zeros(vec![2, 3]);
    assert_eq!(z.numel(), 6);
    assert!(z.to_vec().iter().all(|&v| v == 0.0));

    let o = Tensor::ones(vec![3, 2]);
    assert_eq!(o.numel(), 6);
    assert!(o.to_vec().iter().all(|&v| v == 1.0));

    let f = Tensor::full(vec![2, 2], 3.14);
    assert!(f.to_vec().iter().all(|&v| (v - 3.14).abs() < 1e-6));

    let r = Tensor::randn(vec![4, 4]);
    assert_eq!(r.shape(), &[4, 4]);
}

/// @covers: Tensor ops (tensor-engine dependency)
#[test]
fn test_tensor_ops_exercise_swe_ml_tensor() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    let sum = a.add_raw(&b).unwrap();
    assert_eq!(sum.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);

    let diff = b.sub_raw(&a).unwrap();
    assert_eq!(diff.to_vec(), vec![4.0, 4.0, 4.0, 4.0]);

    let prod = a.mul_raw(&b).unwrap();
    assert_eq!(prod.to_vec(), vec![5.0, 12.0, 21.0, 32.0]);

    let matmul = a.matmul_raw(&b).unwrap();
    assert_eq!(matmul.shape(), &[2, 2]);
}

/// @covers: tape module (no_grad, backward, clear_tape)
#[test]
fn test_tape_no_grad_prevents_recording() {
    tape::clear_tape();
    let result = tape::no_grad(|| {
        assert!(!tape::is_recording());
        42
    });
    assert_eq!(result, 42);
    assert!(tape::is_recording());
}

/// @covers: pool module (acquire, release, clear_pool)
#[test]
fn test_pool_acquire_release_cycle() {
    let buf = pool::acquire(16);
    assert_eq!(buf.len(), 16);
    assert!(buf.iter().all(|&v| v == 0.0));
    pool::release(buf);
    pool::clear_pool();
}

/// @covers: tensor-engine CoreTensor interop
#[test]
fn test_core_tensor_wrapping() {
    let core = CoreTensor::zeros(vec![2, 3]);
    let wrapped = Tensor::new(core, false);
    assert_eq!(wrapped.shape(), &[2, 3]);
    assert!(!wrapped.requires_grad());
}
