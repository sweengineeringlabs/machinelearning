use crate::api::tape::BackwardOp;
use crate::api::tensor::Tensor;

/// Backward for C = A + B (with broadcasting support)
/// saved[0] = shape of A (encoded as tensor), saved[1] = shape of B (encoded as tensor)
/// grad_A = unbroadcast(grad_output, shape_A)
/// grad_B = unbroadcast(grad_output, shape_B)
pub(crate) struct AddBackward {
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl BackwardOp for AddBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        let grad_a = unbroadcast(grad_output, &self.a_shape);
        let grad_b = unbroadcast(grad_output, &self.b_shape);
        vec![grad_a, grad_b]
    }

    fn name(&self) -> &str {
        "AddBackward"
    }
}

/// Reduce gradient back to the original shape when broadcasting occurred.
pub(crate) fn unbroadcast(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return grad.clone();
    }

    let mut result = grad.clone();

    // Sum over leading dimensions that were broadcast (added)
    let extra_dims = grad_shape.len().saturating_sub(target_shape.len());
    for _ in 0..extra_dims {
        result = result.sum_raw(0).expect("unbroadcast sum leading dim");
    }

    // Sum over dimensions that were broadcast from size 1
    let result_shape = result.shape().to_vec();
    for (i, &target_dim) in target_shape.iter().enumerate() {
        if target_dim == 1 && i < result_shape.len() && result_shape[i] != 1 {
            result = result.sum_raw(i as i64).expect("unbroadcast sum dim");
            // Reshape to keep the dimension
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(i, 1);
            result = result.reshape_raw(&new_shape).expect("unbroadcast reshape");
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: AddBackward::backward
    #[test]
    fn test_add_backward_produces_two_gradients() {
        let op = AddBackward {
            a_shape: vec![2, 3],
            b_shape: vec![2, 3],
        };
        let grad = Tensor::ones(vec![2, 3]);
        let grads = op.backward(&grad, &[]);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 3]);
        assert_eq!(grads[1].shape(), &[2, 3]);
    }

    /// @covers: unbroadcast
    #[test]
    fn test_unbroadcast_same_shape_returns_clone() {
        let t = Tensor::ones(vec![2, 3]);
        let result = unbroadcast(&t, &[2, 3]);
        assert_eq!(result.shape(), &[2, 3]);
    }
}
