use swe_ml_tensor::{DType, Tensor as CoreTensor, TensorError, TensorResult};
use crate::api::error::{SwetsError, SwetsResult};
pub use crate::api::tensor_id::TensorId;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) id: TensorId,
    pub(crate) inner: CoreTensor,
    pub(crate) requires_grad: bool,
}

impl Tensor {
    // --- Constructors ---

    pub fn new(inner: CoreTensor, requires_grad: bool) -> Self {
        Self {
            id: TensorId::next(),
            inner,
            requires_grad,
        }
    }

    pub fn zeros(shape: impl Into<swe_ml_tensor::Shape>) -> Self {
        Self::new(CoreTensor::zeros(shape), false)
    }

    pub fn ones(shape: impl Into<swe_ml_tensor::Shape>) -> Self {
        Self::new(CoreTensor::ones(shape), false)
    }

    pub fn randn(shape: impl Into<swe_ml_tensor::Shape>) -> Self {
        Self::new(CoreTensor::randn(shape), false)
    }

    pub fn from_vec(data: Vec<f32>, shape: impl Into<swe_ml_tensor::Shape>) -> TensorResult<Self> {
        Ok(Self::new(CoreTensor::from_vec(data, shape)?, false))
    }

    pub fn full(shape: impl Into<swe_ml_tensor::Shape>, value: f32) -> Self {
        Self::new(CoreTensor::full(shape, value), false)
    }

    // --- Accessors ---

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn inner(&self) -> &CoreTensor {
        &self.inner
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    pub fn data(&self) -> TensorResult<&[f32]> {
        self.inner.data()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.inner.to_vec()
    }

    // --- Raw ops (no tape recording) ---

    pub fn matmul_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.matmul(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn add_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.add(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn sub_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.sub(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn mul_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.mul(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn mul_scalar_raw(&self, scalar: f32) -> Tensor {
        Tensor::new(self.inner.mul_scalar(scalar), false)
    }

    pub fn transpose_raw(&self, dim0: i64, dim1: i64) -> TensorResult<Tensor> {
        let result = self.inner.transpose(dim0, dim1)?;
        Ok(Tensor::new(result, false))
    }

    pub fn relu_raw(&self) -> Tensor {
        Tensor::new(self.inner.relu(), false)
    }

    pub fn neg_raw(&self) -> Tensor {
        Tensor::new(self.inner.neg(), false)
    }

    pub fn mean_all_raw(&self) -> f32 {
        self.inner.mean_all()
    }

    pub fn sum_all_raw(&self) -> f32 {
        self.inner.sum_all()
    }

    pub fn pow_raw(&self, exp: f32) -> Tensor {
        Tensor::new(self.inner.pow(exp), false)
    }

    pub fn div_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.div(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn div_scalar_raw(&self, scalar: f32) -> Tensor {
        Tensor::new(self.inner.div_scalar(scalar), false)
    }

    pub fn add_scalar_raw(&self, scalar: f32) -> Tensor {
        Tensor::new(self.inner.add_scalar(scalar), false)
    }

    pub fn sqrt_raw(&self) -> Tensor {
        Tensor::new(self.inner.sqrt(), false)
    }

    pub fn reshape_raw(&self, shape: &[usize]) -> TensorResult<Tensor> {
        let result = self.inner.reshape(shape)?;
        Ok(Tensor::new(result, false))
    }

    pub fn sum_raw(&self, dim: i64) -> TensorResult<Tensor> {
        let result = self.inner.sum(dim)?;
        Ok(Tensor::new(result, false))
    }

    // --- FR-106: Shape ops (no tape recording) ---

    /// Permute dimensions. Delegates to CoreTensor::permute (zero-copy stride reorder).
    pub fn permute_raw(&self, dims: &[usize]) -> SwetsResult<Tensor> {
        let result = self.inner.permute(dims).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Remove a dimension of size 1. Delegates to CoreTensor::squeeze.
    pub fn squeeze_raw(&self, dim: i64) -> SwetsResult<Tensor> {
        let result = self.inner.squeeze(dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Insert a dimension of size 1. Delegates to CoreTensor::unsqueeze.
    pub fn unsqueeze_raw(&self, dim: i64) -> SwetsResult<Tensor> {
        let result = self.inner.unsqueeze(dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Flatten the tensor into a 1D tensor.
    /// Implemented via reshape to a single dimension equal to numel().
    pub fn flatten_raw(&self) -> SwetsResult<Tensor> {
        let n = self.numel();
        let result = self.inner.reshape(&[n]).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// View the tensor with a new shape (alias for reshape).
    /// Total number of elements must remain the same.
    pub fn view_raw(&self, shape: &[usize]) -> SwetsResult<Tensor> {
        let result = self.inner.reshape(shape).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    // --- FR-107: Indexing ops (no tape recording) ---

    /// Slice the tensor along a dimension from `start` to `end` (exclusive).
    /// Delegates to CoreTensor::slice.
    pub fn slice_raw(&self, dim: i64, start: usize, end: usize) -> SwetsResult<Tensor> {
        let result = self.inner.slice(dim, start, end).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Select elements along `dim` at the positions specified by `indices`.
    /// Returns a tensor with the same number of dimensions where `dim` has size `indices.len()`.
    ///
    /// CoreTensor does not provide index_select natively, so this is implemented
    /// element-wise via to_vec()/from_vec().
    pub fn index_select_raw(&self, dim: i64, indices: &[usize]) -> SwetsResult<Tensor> {
        let shape = self.shape();
        let ndim = self.ndim();
        let dim_idx = normalize_dim_helper(dim, ndim)?;
        let dim_size = shape[dim_idx];

        // Validate indices
        for &idx in indices {
            if idx >= dim_size {
                return Err(SwetsError::TensorError(TensorError::IndexOutOfBounds {
                    dim: dim_idx,
                    index: idx,
                    size: dim_size,
                }));
            }
        }

        // Compute output shape
        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape[dim_idx] = indices.len();
        let out_numel: usize = out_shape.iter().product();

        let src_data = self.to_vec();
        let src_strides = compute_strides(&shape);
        let out_strides = compute_strides(&out_shape);

        let mut out_data = vec![0.0f32; out_numel];
        let mut out_indices = vec![0usize; ndim];

        for flat in 0..out_numel {
            // Convert flat index to multi-dimensional index in output
            let mut rem = flat;
            for d in 0..ndim {
                out_indices[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }
            // Map the output index to source index
            let mut src_flat = 0;
            for d in 0..ndim {
                let idx = if d == dim_idx {
                    indices[out_indices[d]]
                } else {
                    out_indices[d]
                };
                src_flat += idx * src_strides[d];
            }
            out_data[flat] = src_data[src_flat];
        }

        let result = CoreTensor::from_vec(out_data, out_shape).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Gather elements along `dim` using the index tensor.
    /// `index` must have the same number of dimensions as `self`.
    /// The output has the same shape as `index`.
    ///
    /// For a 3D tensor: `output[i][j][k] = self[i][index[i][j][k]][k]` when dim=1.
    ///
    /// CoreTensor does not provide gather natively, so this is implemented
    /// element-wise via to_vec()/from_vec().
    pub fn gather_raw(&self, dim: i64, index: &Tensor) -> SwetsResult<Tensor> {
        let shape = self.shape();
        let ndim = self.ndim();
        let dim_idx = normalize_dim_helper(dim, ndim)?;

        let idx_shape = index.shape();
        if idx_shape.len() != ndim {
            return Err(SwetsError::TensorError(TensorError::InvalidOperation(
                format!(
                    "gather: index must have same ndim as self ({}), got {}",
                    ndim,
                    idx_shape.len()
                ),
            )));
        }

        let src_data = self.to_vec();
        let idx_data = index.to_vec();
        let src_strides = compute_strides(shape);
        let idx_strides = compute_strides(idx_shape);
        let out_numel: usize = idx_shape.iter().product();

        let mut out_data = vec![0.0f32; out_numel];
        let mut multi_idx = vec![0usize; ndim];

        for flat in 0..out_numel {
            // Convert flat index to multi-dimensional index in index tensor
            let mut rem = flat;
            for d in 0..ndim {
                multi_idx[d] = rem / idx_strides[d];
                rem %= idx_strides[d];
            }
            // The gather index along dim
            let gather_idx = idx_data[flat] as usize;
            if gather_idx >= shape[dim_idx] {
                return Err(SwetsError::TensorError(TensorError::IndexOutOfBounds {
                    dim: dim_idx,
                    index: gather_idx,
                    size: shape[dim_idx],
                }));
            }
            // Compute source flat index
            let mut src_flat = 0;
            for d in 0..ndim {
                let idx = if d == dim_idx { gather_idx } else { multi_idx[d] };
                src_flat += idx * src_strides[d];
            }
            out_data[flat] = src_data[src_flat];
        }

        let result =
            CoreTensor::from_vec(out_data, idx_shape.to_vec()).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Select elements where `mask` is non-zero. Returns a 1D tensor.
    /// `mask` must be broadcastable to `self`'s shape.
    ///
    /// CoreTensor does not provide masked_select natively, so this is implemented
    /// element-wise via to_vec()/from_vec().
    pub fn masked_select_raw(&self, mask: &Tensor) -> SwetsResult<Tensor> {
        let self_data = self.to_vec();

        // Broadcast mask to self's shape if needed
        let mask_data = if self.shape() == mask.shape() {
            mask.to_vec()
        } else {
            let target = swe_ml_tensor::Shape::new(self.shape().to_vec());
            let broadcast_mask = mask
                .inner
                .broadcast_to(&target)
                .map_err(SwetsError::TensorError)?;
            broadcast_mask.to_vec()
        };

        let selected: Vec<f32> = self_data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, m)| **m != 0.0)
            .map(|(v, _)| *v)
            .collect();
        let n = selected.len();
        let result = CoreTensor::from_vec(selected, vec![n]).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    // --- FR-108: Element-wise math ops (no tape recording) ---

    /// Element-wise exponential. Delegates to CoreTensor::exp.
    pub fn exp_raw(&self) -> Tensor {
        Tensor::new(self.inner.exp(), false)
    }

    /// Element-wise natural logarithm. Delegates to CoreTensor::log (which calls ln()).
    pub fn log_raw(&self) -> Tensor {
        Tensor::new(self.inner.log(), false)
    }

    /// Element-wise absolute value. Delegates to CoreTensor::abs.
    pub fn abs_raw(&self) -> Tensor {
        Tensor::new(self.inner.abs(), false)
    }

    // --- FR-109: Reduction ops (no tape recording) ---

    /// Mean along a dimension. Delegates to CoreTensor::mean.
    pub fn mean_raw(&self, dim: i64) -> SwetsResult<Tensor> {
        let result = self.inner.mean(dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Max along a dimension. Returns (values, indices).
    /// Delegates to CoreTensor::max.
    pub fn max_raw(&self, dim: i64) -> SwetsResult<(Tensor, Tensor)> {
        let (values, indices) = self.inner.max(dim).map_err(SwetsError::TensorError)?;
        Ok((Tensor::new(values, false), Tensor::new(indices, false)))
    }

    /// Min along a dimension. Returns (values, indices).
    /// Delegates to CoreTensor::min.
    pub fn min_raw(&self, dim: i64) -> SwetsResult<(Tensor, Tensor)> {
        let (values, indices) = self.inner.min(dim).map_err(SwetsError::TensorError)?;
        Ok((Tensor::new(values, false), Tensor::new(indices, false)))
    }

    /// Variance along a dimension. Delegates to CoreTensor::var.
    pub fn var_raw(&self, dim: i64) -> SwetsResult<Tensor> {
        let result = self.inner.var(dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Standard deviation along a dimension.
    /// Computed as sqrt(var(dim)).
    pub fn std_dev_raw(&self, dim: i64) -> SwetsResult<Tensor> {
        let var = self.inner.var(dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(var.sqrt(), false))
    }

    // --- FR-112: Joining ops (no tape recording) ---

    /// Concatenate tensors along a dimension. Delegates to CoreTensor::cat.
    pub fn concat_raw(tensors: &[&Tensor], dim: i64) -> SwetsResult<Tensor> {
        let core_tensors: Vec<&CoreTensor> = tensors.iter().map(|t| &t.inner).collect();
        let result = CoreTensor::cat(&core_tensors, dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Stack tensors along a new dimension.
    /// All tensors must have the same shape. A new dimension of size `tensors.len()`
    /// is inserted at position `dim`.
    ///
    /// CoreTensor does not provide stack natively, so this is implemented by
    /// unsqueezing each tensor and then concatenating.
    pub fn stack_raw(tensors: &[&Tensor], dim: i64) -> SwetsResult<Tensor> {
        if tensors.is_empty() {
            return Err(SwetsError::TensorError(TensorError::EmptyTensor));
        }

        let first_shape = tensors[0].shape();
        for t in tensors.iter().skip(1) {
            if t.shape() != first_shape {
                return Err(SwetsError::TensorError(TensorError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    got: t.shape().to_vec(),
                }));
            }
        }

        // Unsqueeze each tensor at the target dim, then concatenate
        let unsqueezed: Vec<CoreTensor> = tensors
            .iter()
            .map(|t| t.inner.unsqueeze(dim))
            .collect::<Result<Vec<_>, _>>()
            .map_err(SwetsError::TensorError)?;
        let refs: Vec<&CoreTensor> = unsqueezed.iter().collect();
        let result = CoreTensor::cat(&refs, dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Split a tensor into chunks along a dimension.
    /// `split_size` is the size of each chunk (the last chunk may be smaller).
    ///
    /// CoreTensor does not provide split natively, so this is implemented via
    /// repeated slicing.
    pub fn split_raw(&self, split_size: usize, dim: i64) -> SwetsResult<Vec<Tensor>> {
        let shape = self.shape();
        let ndim = self.ndim();
        let dim_idx = normalize_dim_helper(dim, ndim)?;
        let dim_size = shape[dim_idx];

        if split_size == 0 {
            return Err(SwetsError::TensorError(TensorError::InvalidOperation(
                "split_size must be greater than 0".into(),
            )));
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        while start < dim_size {
            let end = (start + split_size).min(dim_size);
            let chunk = self
                .inner
                .slice(dim, start, end)
                .map_err(SwetsError::TensorError)?;
            chunks.push(Tensor::new(chunk, false));
            start = end;
        }

        Ok(chunks)
    }

    /// Replace inner data while preserving TensorId (for optimizer in-place updates).
    pub fn update_data_from(&mut self, other: &Tensor) {
        self.inner = other.inner.clone();
    }
}

// --- Helper functions ---

/// Normalize a dimension index, converting negative indices to positive.
fn normalize_dim_helper(dim: i64, ndim: usize) -> SwetsResult<usize> {
    let ndim_i64 = ndim as i64;
    let normalized = if dim < 0 { dim + ndim_i64 } else { dim };
    if normalized >= 0 && normalized < ndim_i64 {
        Ok(normalized as usize)
    } else {
        Err(SwetsError::TensorError(TensorError::InvalidDimension {
            dim,
            ndim,
        }))
    }
}

/// Compute row-major strides from a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: Tensor::zeros
    #[test]
    fn test_zeros_creates_all_zero_tensor() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.to_vec().iter().all(|&v| v == 0.0));
    }

    /// @covers: Tensor::ones
    #[test]
    fn test_ones_creates_all_one_tensor() {
        let t = Tensor::ones(vec![2, 3]);
        assert!(t.to_vec().iter().all(|&v| v == 1.0));
    }

    /// @covers: Tensor::numel
    #[test]
    fn test_numel_returns_total_elements() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.numel(), 24);
    }

    /// @covers: Tensor::add_raw
    #[test]
    fn test_add_raw_sums_element_wise() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
        let c = a.add_raw(&b).unwrap();
        assert_eq!(c.to_vec(), vec![4.0, 6.0]);
    }

    /// @covers: Tensor::mul_scalar_raw
    #[test]
    fn test_mul_scalar_raw_scales_all_elements() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = a.mul_scalar_raw(2.0);
        assert_eq!(b.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    /// @covers: TensorId::next
    #[test]
    fn test_tensor_ids_are_unique() {
        let t1 = Tensor::zeros(vec![1]);
        let t2 = Tensor::zeros(vec![1]);
        assert_ne!(t1.id(), t2.id());
    }

    /// @covers: Tensor::set_requires_grad
    #[test]
    fn test_set_requires_grad_toggles_flag() {
        let mut t = Tensor::zeros(vec![2]);
        assert!(!t.requires_grad());
        t.set_requires_grad(true);
        assert!(t.requires_grad());
    }

    /// @covers: compute_strides
    #[test]
    fn test_compute_strides_for_3d_shape() {
        let strides = compute_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    /// @covers: compute_strides
    #[test]
    fn test_compute_strides_empty_shape() {
        let strides = compute_strides(&[]);
        assert!(strides.is_empty());
    }

    /// @covers: normalize_dim_helper
    #[test]
    fn test_normalize_dim_helper_negative_dim() {
        let result = normalize_dim_helper(-1, 3).unwrap();
        assert_eq!(result, 2);
    }

    /// @covers: normalize_dim_helper
    #[test]
    fn test_normalize_dim_helper_out_of_bounds() {
        let result = normalize_dim_helper(5, 3);
        assert!(result.is_err());
    }

    /// @covers: Tensor::randn
    #[test]
    fn test_randn_creates_tensor_with_correct_shape() {
        let t = Tensor::randn(vec![3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.numel(), 12);
    }

    /// @covers: Tensor::from_vec
    #[test]
    fn test_from_vec_creates_tensor_with_data() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    /// @covers: Tensor::full
    #[test]
    fn test_full_creates_tensor_with_value() {
        let t = Tensor::full(vec![2, 3], 5.0);
        assert!(t.to_vec().iter().all(|&v| (v - 5.0).abs() < 1e-6));
    }

    /// @covers: Tensor::inner
    #[test]
    fn test_inner_returns_core_tensor() {
        let t = Tensor::zeros(vec![2, 3]);
        let inner = t.inner();
        assert_eq!(inner.shape(), &[2, 3]);
    }

    /// @covers: Tensor::ndim
    #[test]
    fn test_ndim_returns_number_of_dimensions() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.ndim(), 3);
    }

    /// @covers: Tensor::dtype
    #[test]
    fn test_dtype_returns_tensor_dtype() {
        let t = Tensor::zeros(vec![2]);
        let _ = t.dtype(); // Just ensure it doesn't panic
    }

    /// @covers: Tensor::data
    #[test]
    fn test_data_returns_slice() {
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let data = t.data().unwrap();
        assert_eq!(data, &[1.0, 2.0]);
    }

    /// @covers: Tensor::to_vec
    #[test]
    fn test_to_vec_returns_owned_data() {
        let t = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
        assert_eq!(t.to_vec(), vec![3.0, 4.0]);
    }

    /// @covers: Tensor::new
    #[test]
    fn test_new_wraps_core_tensor() {
        let core = swe_ml_tensor::Tensor::zeros(vec![2]);
        let t = Tensor::new(core, true);
        assert!(t.requires_grad());
        assert_eq!(t.shape(), &[2]);
    }

    /// @covers: Tensor::id
    #[test]
    fn test_id_returns_unique_id() {
        let t = Tensor::zeros(vec![1]);
        let _ = t.id();
    }

    /// @covers: Tensor::shape
    #[test]
    fn test_shape_returns_dimensions() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
    }

    /// @covers: Tensor::requires_grad
    #[test]
    fn test_requires_grad_returns_false_by_default() {
        let t = Tensor::zeros(vec![2]);
        assert!(!t.requires_grad());
    }

    /// @covers: Tensor::matmul_raw
    #[test]
    fn test_matmul_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c = a.matmul_raw(&b).unwrap();
        assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// @covers: Tensor::sub_raw
    #[test]
    fn test_sub_raw() {
        let a = Tensor::from_vec(vec![5.0, 3.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![2.0, 1.0], vec![2]).unwrap();
        let c = a.sub_raw(&b).unwrap();
        assert_eq!(c.to_vec(), vec![3.0, 2.0]);
    }

    /// @covers: Tensor::mul_raw
    #[test]
    fn test_mul_raw() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]).unwrap();
        let c = a.mul_raw(&b).unwrap();
        assert_eq!(c.to_vec(), vec![8.0, 15.0]);
    }

    /// @covers: Tensor::transpose_raw
    #[test]
    fn test_transpose_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let t = a.transpose_raw(0, 1).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
    }

    /// @covers: Tensor::relu_raw
    #[test]
    fn test_relu_raw() {
        let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]).unwrap();
        let r = a.relu_raw();
        assert_eq!(r.to_vec(), vec![0.0, 0.0, 1.0]);
    }

    /// @covers: Tensor::neg_raw
    #[test]
    fn test_neg_raw() {
        let a = Tensor::from_vec(vec![1.0, -2.0], vec![2]).unwrap();
        let n = a.neg_raw();
        assert_eq!(n.to_vec(), vec![-1.0, 2.0]);
    }

    /// @covers: Tensor::mean_all_raw
    #[test]
    fn test_mean_all_raw() {
        let a = Tensor::from_vec(vec![2.0, 4.0], vec![2]).unwrap();
        assert!((a.mean_all_raw() - 3.0).abs() < 1e-6);
    }

    /// @covers: Tensor::sum_all_raw
    #[test]
    fn test_sum_all_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert!((a.sum_all_raw() - 6.0).abs() < 1e-6);
    }

    /// @covers: Tensor::pow_raw
    #[test]
    fn test_pow_raw() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]).unwrap();
        let p = a.pow_raw(2.0);
        assert_eq!(p.to_vec(), vec![4.0, 9.0]);
    }

    /// @covers: Tensor::div_raw
    #[test]
    fn test_div_raw() {
        let a = Tensor::from_vec(vec![6.0, 8.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![2.0, 4.0], vec![2]).unwrap();
        let c = a.div_raw(&b).unwrap();
        assert_eq!(c.to_vec(), vec![3.0, 2.0]);
    }

    /// @covers: Tensor::div_scalar_raw
    #[test]
    fn test_div_scalar_raw() {
        let a = Tensor::from_vec(vec![4.0, 6.0], vec![2]).unwrap();
        let c = a.div_scalar_raw(2.0);
        assert_eq!(c.to_vec(), vec![2.0, 3.0]);
    }

    /// @covers: Tensor::add_scalar_raw
    #[test]
    fn test_add_scalar_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let c = a.add_scalar_raw(3.0);
        assert_eq!(c.to_vec(), vec![4.0, 5.0]);
    }

    /// @covers: Tensor::sqrt_raw
    #[test]
    fn test_sqrt_raw() {
        let a = Tensor::from_vec(vec![4.0, 9.0], vec![2]).unwrap();
        let s = a.sqrt_raw();
        assert!((s.to_vec()[0] - 2.0).abs() < 1e-6);
        assert!((s.to_vec()[1] - 3.0).abs() < 1e-6);
    }

    /// @covers: Tensor::reshape_raw
    #[test]
    fn test_reshape_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let r = a.reshape_raw(&[4]).unwrap();
        assert_eq!(r.shape(), &[4]);
    }

    /// @covers: Tensor::sum_raw
    #[test]
    fn test_sum_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let s = a.sum_raw(1).unwrap();
        assert_eq!(s.to_vec(), vec![3.0, 7.0]);
    }

    /// @covers: Tensor::permute_raw
    #[test]
    fn test_permute_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let p = a.permute_raw(&[1, 0]).unwrap();
        assert_eq!(p.shape(), &[3, 2]);
    }

    /// @covers: Tensor::squeeze_raw
    #[test]
    fn test_squeeze_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let s = a.squeeze_raw(0).unwrap();
        assert_eq!(s.shape(), &[2]);
    }

    /// @covers: Tensor::unsqueeze_raw
    #[test]
    fn test_unsqueeze_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let u = a.unsqueeze_raw(0).unwrap();
        assert_eq!(u.shape(), &[1, 2]);
    }

    /// @covers: Tensor::flatten_raw
    #[test]
    fn test_flatten_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let f = a.flatten_raw().unwrap();
        assert_eq!(f.shape(), &[4]);
    }

    /// @covers: Tensor::view_raw
    #[test]
    fn test_view_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let v = a.view_raw(&[3, 2]).unwrap();
        assert_eq!(v.shape(), &[3, 2]);
    }

    /// @covers: Tensor::slice_raw
    #[test]
    fn test_slice_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let s = a.slice_raw(0, 1, 3).unwrap();
        assert_eq!(s.to_vec(), vec![2.0, 3.0]);
    }

    /// @covers: Tensor::exp_raw
    #[test]
    fn test_exp_raw() {
        let a = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let e = a.exp_raw();
        assert!((e.to_vec()[0] - 1.0).abs() < 1e-6);
    }

    /// @covers: Tensor::log_raw
    #[test]
    fn test_log_raw() {
        let a = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        let l = a.log_raw();
        assert!(l.to_vec()[0].abs() < 1e-6);
    }

    /// @covers: Tensor::abs_raw
    #[test]
    fn test_abs_raw() {
        let a = Tensor::from_vec(vec![-1.0, 2.0, -3.0], vec![3]).unwrap();
        let ab = a.abs_raw();
        assert_eq!(ab.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    /// @covers: Tensor::mean_raw
    #[test]
    fn test_mean_raw() {
        let a = Tensor::from_vec(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let m = a.mean_raw(1).unwrap();
        assert!((m.to_vec()[0] - 2.0).abs() < 1e-6);
        assert!((m.to_vec()[1] - 3.0).abs() < 1e-6);
    }

    /// @covers: Tensor::max_raw
    #[test]
    fn test_max_raw() {
        let a = Tensor::from_vec(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let (vals, _idxs) = a.max_raw(1).unwrap();
        assert_eq!(vals.to_vec(), vec![3.0, 4.0]);
    }

    /// @covers: Tensor::min_raw
    #[test]
    fn test_min_raw() {
        let a = Tensor::from_vec(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let (vals, _idxs) = a.min_raw(1).unwrap();
        assert_eq!(vals.to_vec(), vec![1.0, 2.0]);
    }

    /// @covers: Tensor::var_raw
    #[test]
    fn test_var_raw() {
        let a = Tensor::from_vec(vec![2.0, 4.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let v = a.var_raw(1).unwrap();
        assert!(v.to_vec()[0] > 0.0);
    }

    /// @covers: Tensor::std_dev_raw
    #[test]
    fn test_std_dev_raw() {
        let a = Tensor::from_vec(vec![2.0, 4.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let s = a.std_dev_raw(1).unwrap();
        assert!(s.to_vec()[0] > 0.0);
    }

    /// @covers: Tensor::concat_raw
    #[test]
    fn test_concat_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
        let c = Tensor::concat_raw(&[&a, &b], 0).unwrap();
        assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// @covers: Tensor::stack_raw
    #[test]
    fn test_stack_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
        let s = Tensor::stack_raw(&[&a, &b], 0).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
    }

    /// @covers: Tensor::split_raw
    #[test]
    fn test_split_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let chunks = a.split_raw(2, 0).unwrap();
        assert_eq!(chunks.len(), 2);
    }

    /// @covers: Tensor::update_data_from
    #[test]
    fn test_update_data_from() {
        let mut a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![5.0, 6.0], vec![2]).unwrap();
        a.update_data_from(&b);
        assert_eq!(a.to_vec(), vec![5.0, 6.0]);
    }

    /// @covers: Tensor::index_select_raw
    #[test]
    fn test_index_select_raw() {
        let a = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let s = a.index_select_raw(0, &[0, 2]).unwrap();
        assert_eq!(s.to_vec(), vec![10.0, 30.0]);
    }

    /// @covers: Tensor::gather_raw
    #[test]
    fn test_gather_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let idx = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).unwrap();
        let g = a.gather_raw(1, &idx).unwrap();
        assert_eq!(g.shape(), &[2, 2]);
    }

    /// @covers: Tensor::masked_select_raw
    #[test]
    fn test_masked_select_raw() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mask = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], vec![4]).unwrap();
        let s = a.masked_select_raw(&mask).unwrap();
        assert_eq!(s.to_vec(), vec![1.0, 3.0]);
    }
}
