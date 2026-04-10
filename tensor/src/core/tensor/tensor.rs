//! Tensor implementation methods.
//!
//! The Tensor struct is defined in api/tensor_def.rs.
//! This module provides all impl blocks.

pub use crate::api::tensor::def::Tensor;
pub use crate::api::storage_def::{Storage, f32_vec_to_bytes, f32_slice_to_bytes};
pub(crate) use crate::api::tensor::def::TensorShape;
pub(crate) use crate::api::storage_def::storage_byte_len;
use crate::api::error::{TensorError, TensorResult};
use crate::api::dtype::DType;
use crate::api::device::Device;
use crate::api::traits::TensorOps;
use crate::core::shape_mod::shape::Shape;
use bytemuck;
use half::{bf16, f16};
use rand::Rng;
use smallvec::SmallVec;
use std::fmt;
use std::sync::Arc;

impl Tensor {
    // ==================== Low-level constructors ====================

    /// Create a tensor from raw bytes with the given shape and dtype.
    pub fn new(data: Vec<u8>, shape: impl Into<SmallVec<[usize; 4]>>, dtype: DType) -> Self {
        let shape_sv: TensorShape = shape.into();
        let strides = Self::compute_strides_sv(&shape_sv);
        Self {
            data: Arc::new(Storage::Owned(data)),
            shape_sv,
            strides,
            dtype,
            device: Device::Cpu,
        }
    }

    /// Create a tensor backed by a memory-mapped file region.
    pub fn from_mmap(
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        len: usize,
        shape: impl Into<SmallVec<[usize; 4]>>,
        dtype: DType,
    ) -> Self {
        let shape_sv: TensorShape = shape.into();
        let strides = Self::compute_strides_sv(&shape_sv);
        Self {
            data: Arc::new(Storage::MMap { mmap, offset, len }),
            shape_sv,
            strides,
            dtype,
            device: Device::Cpu,
        }
    }

    /// Create a view with new shape/strides over existing data.
    pub(crate) fn view(
        data: Arc<Storage>,
        shape: TensorShape,
        strides: TensorShape,
        dtype: DType,
    ) -> TensorResult<Self> {
        if !shape.is_empty() && dtype.size() > 0 {
            let mut max_offset: usize = 0;
            for i in 0..shape.len() {
                if shape[i] > 0 {
                    max_offset += (shape[i] - 1) * strides[i];
                }
            }
            let required_bytes = (max_offset + 1) * dtype.size();
            let available = storage_byte_len(&data);
            if required_bytes > available {
                return Err(TensorError::IndexOutOfBounds {
                    dim: 0,
                    index: required_bytes,
                    size: available,
                });
            }
        }
        Ok(Self {
            data,
            shape_sv: shape,
            strides,
            dtype,
            device: Device::Cpu,
        })
    }

    // ==================== High-level constructors (f32) ====================

    /// Create a tensor from an f32 vector with the given shape.
    pub fn from_vec(data: Vec<f32>, shape: impl Into<Shape>) -> TensorResult<Self> {
        let shape = shape.into();
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: shape.dims().to_vec(),
                got: vec![data.len()],
            });
        }
        let shape_sv: TensorShape = SmallVec::from_slice(shape.dims());
        let strides = Self::compute_strides_sv(&shape_sv);
        let bytes = f32_vec_to_bytes(data);
        Ok(Self {
            data: Arc::new(Storage::Owned(bytes)),
            shape_sv,
            strides,
            dtype: DType::F32,
            device: Device::Cpu,
        })
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let n = shape.numel();
        let bytes = vec![0u8; n * 4];
        let shape_sv: TensorShape = SmallVec::from_slice(shape.dims());
        let strides = Self::compute_strides_sv(&shape_sv);
        Self {
            data: Arc::new(Storage::Owned(bytes)),
            shape_sv,
            strides,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let n = shape.numel();
        let data = vec![1.0f32; n];
        let bytes = f32_vec_to_bytes(data);
        let shape_sv: TensorShape = SmallVec::from_slice(shape.dims());
        let strides = Self::compute_strides_sv(&shape_sv);
        Self {
            data: Arc::new(Storage::Owned(bytes)),
            shape_sv,
            strides,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Create a tensor filled with a specific value.
    pub fn full(shape: impl Into<Shape>, value: f32) -> Self {
        let shape = shape.into();
        let n = shape.numel();
        let data = vec![value; n];
        let bytes = f32_vec_to_bytes(data);
        let shape_sv: TensorShape = SmallVec::from_slice(shape.dims());
        let strides = Self::compute_strides_sv(&shape_sv);
        Self {
            data: Arc::new(Storage::Owned(bytes)),
            shape_sv,
            strides,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Create a tensor with random values from standard normal distribution.
    ///
    /// Uses ziggurat algorithm via `rand_distr::StandardNormal` — no
    /// transcendental functions (ln, cos, sin), ~10x faster than Box-Muller.
    pub fn randn(shape: impl Into<Shape>) -> Self {
        use rand_distr::{Distribution, StandardNormal};

        let shape = shape.into();
        let numel = shape.numel();
        let mut rng = rand::thread_rng();

        // Allocate f32 vec directly — no zeroing, no byte conversion
        let mut data: Vec<f32> = Vec::with_capacity(numel);
        // SAFETY: we fill every element below before anyone reads
        unsafe { data.set_len(numel); }
        for v in data.iter_mut() {
            *v = StandardNormal.sample(&mut rng);
        }

        let bytes = f32_vec_to_bytes(data);
        let shape_sv: TensorShape = SmallVec::from_slice(shape.dims());
        let strides = Self::compute_strides_sv(&shape_sv);
        Self {
            data: Arc::new(Storage::Owned(bytes)),
            shape_sv,
            strides,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Create a tensor with random uniform values in [0, 1).
    pub fn rand(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..shape.numel()).map(|_| rng.r#gen()).collect();
        let bytes = f32_vec_to_bytes(data);
        let shape_sv: TensorShape = SmallVec::from_slice(shape.dims());
        let strides = Self::compute_strides_sv(&shape_sv);
        Self {
            data: Arc::new(Storage::Owned(bytes)),
            shape_sv,
            strides,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Create an identity matrix.
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::from_vec(data, vec![n, n]).unwrap()
    }

    /// Create a lower triangular matrix of ones.
    pub fn tril(size: usize) -> Self {
        let mut data = vec![0.0f32; size * size];
        for i in 0..size {
            for j in 0..=i {
                data[i * size + j] = 1.0;
            }
        }
        Self::from_vec(data, vec![size, size]).unwrap()
    }

    /// Create an upper triangular matrix of ones.
    pub fn triu(size: usize) -> Self {
        let mut data = vec![0.0f32; size * size];
        for i in 0..size {
            for j in i..size {
                data[i * size + j] = 1.0;
            }
        }
        Self::from_vec(data, vec![size, size]).unwrap()
    }

    /// Create a 1D tensor with values from start to end (exclusive).
    pub fn arange(start: f32, end: f32, step: f32) -> TensorResult<Self> {
        if step == 0.0 {
            return Err(TensorError::InvalidOperation("step cannot be zero".into()));
        }
        let n = ((end - start) / step).ceil() as usize;
        let data: Vec<f32> = (0..n).map(|i| start + (i as f32) * step).collect();
        Self::from_vec(data, vec![n])
    }

    /// Create an empty tensor.
    pub fn empty() -> Self {
        Self::new(vec![], SmallVec::from_slice(&[0usize]), DType::F32)
    }

    /// Create a new tensor with aligned storage (for SIMD).
    pub fn new_aligned(shape: impl Into<SmallVec<[usize; 4]>>, dtype: DType) -> Self {
        let shape_sv: TensorShape = shape.into();
        let n_elements: usize = shape_sv.iter().product();
        let byte_size = if dtype.size() > 0 {
            n_elements * dtype.size()
        } else {
            n_elements * 4
        };
        let data = vec![0u8; byte_size];
        let strides = Self::compute_strides_sv(&shape_sv);
        Self {
            data: Arc::new(Storage::Owned(data)),
            shape_sv,
            strides,
            dtype,
            device: Device::Cpu,
        }
    }

    // ==================== Properties ====================

    /// Get the shape as a slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape_sv
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape_sv.len()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape_sv.iter().product()
    }

    /// Alias for numel().
    pub fn element_count(&self) -> usize {
        self.numel()
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the underlying f32 data as a slice (contiguous F32 tensors only).
    pub fn data(&self) -> TensorResult<&[f32]> {
        self.as_slice_f32()
    }

    /// Get the underlying f32 data as a slice.
    pub fn as_slice_f32(&self) -> TensorResult<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::F32,
                got: self.dtype,
            });
        }
        let bytes = self.as_raw_bytes()?;
        if bytes.len() % 4 != 0 {
            return Err(TensorError::ConversionError(
                "byte length not divisible by 4".into(),
            ));
        }
        bytemuck::try_cast_slice(bytes).map_err(|_| {
            TensorError::ConversionError(
                "as_slice_f32: data not 4-byte aligned; call to_f32() first".into(),
            )
        })
    }

    /// Get mutable access to the underlying f32 data.
    ///
    /// If the storage is shared (Arc refcount > 1) or not Owned, this will
    /// copy the data first to ensure unique ownership.
    pub fn as_mut_slice_f32(&mut self) -> TensorResult<&mut [f32]> {
        if self.dtype != DType::F32 {
            return Err(TensorError::DTypeMismatch {
                expected: DType::F32,
                got: self.dtype,
            });
        }
        // Try to get mutable access without copying
        if Arc::get_mut(&mut self.data)
            .and_then(|s| match s {
                Storage::Owned(_) => Some(()),
                _ => None,
            })
            .is_none()
        {
            // Copy data to get unique ownership
            let bytes = self.as_raw_bytes()?.to_vec();
            self.data = Arc::new(Storage::Owned(bytes));
        }
        // Now we have unique Owned storage
        match Arc::get_mut(&mut self.data).unwrap() {
            Storage::Owned(v) => bytemuck::try_cast_slice_mut(v.as_mut_slice()).map_err(
                |_| TensorError::ConversionError("as_mut_slice_f32: data not 4-byte aligned".into()),
            ),
            _ => unreachable!(),
        }
    }

    /// Returns raw byte slice from any Storage variant.
    pub fn as_raw_bytes(&self) -> TensorResult<&[u8]> {
        match self.data.as_ref() {
            Storage::Owned(v) => Ok(v.as_slice()),
            Storage::View { parent, offset, len } => {
                let parent_bytes = parent.as_raw_bytes()?;
                Ok(&parent_bytes[*offset..*offset + *len])
            }
            Storage::MMap { mmap, offset, len } => Ok(&mmap[*offset..*offset + *len]),
        }
    }

    /// Returns a raw pointer into the underlying storage.
    ///
    /// # Safety
    /// The caller must ensure the returned pointer is only used while `self` is alive
    /// and the pointed-to range is within bounds.
    pub(crate) unsafe fn as_ptr(&self) -> TensorResult<*const u8> {
        match self.data.as_ref() {
            Storage::Owned(v) => Ok(v.as_ptr()),
            Storage::View { parent, offset, .. } => {
                Ok(unsafe { parent.as_ptr()?.add(*offset) })
            }
            Storage::MMap { mmap, offset, .. } => {
                Ok(unsafe { mmap.as_ptr().add(*offset) })
            }
        }
    }

    /// Convert to a Vec<f32> (always works for F32 tensors, may copy).
    pub fn to_vec(&self) -> Vec<f32> {
        if self.dtype == DType::F32 {
            if self.is_contiguous() {
                if let Ok(slice) = self.as_slice_f32() {
                    return slice.to_vec();
                }
            }
            // Non-contiguous: use iterator
            return self.iter().collect();
        }
        // For non-F32: convert first
        if let Ok(f32_tensor) = self.to_f32() {
            if let Ok(slice) = f32_tensor.as_slice_f32() {
                return slice.to_vec();
            }
        }
        vec![]
    }

    /// Iterate over all elements as f32.
    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        TensorIterator::new(self)
    }

    /// Check if tensor is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        let default_strides = Self::compute_strides_sv(&self.shape_sv);
        self.strides == default_strides
    }

    /// Convert to F32 tensor (handles Q4_0, Q8_0, F16, BF16 dequantization).
    pub fn to_f32(&self) -> TensorResult<Tensor> {
        match self.dtype {
            DType::F32 => {
                let bytes = self.as_raw_bytes()?;
                if bytemuck::try_cast_slice::<u8, f32>(bytes).is_ok() {
                    Ok(self.clone())
                } else {
                    Ok(Tensor::new(
                        bytes.to_vec(),
                        self.shape_sv.clone(),
                        DType::F32,
                    ))
                }
            }
            DType::BF16 => {
                let contiguous = if !self.is_contiguous() {
                    self.contiguous()?
                } else {
                    self.clone()
                };
                let bytes = contiguous.as_raw_bytes()?;
                let data_f32: Vec<f32> = match bytemuck::try_cast_slice::<u8, bf16>(bytes) {
                    Ok(slice) => slice.iter().map(|x| x.to_f32()).collect(),
                    Err(_) => {
                        let n = bytes.len() / 2;
                        (0..n)
                            .map(|i| {
                                bf16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]).to_f32()
                            })
                            .collect()
                    }
                };
                Ok(Tensor::new(
                    f32_vec_to_bytes(data_f32),
                    contiguous.shape_sv.clone(),
                    DType::F32,
                ))
            }
            DType::F16 => {
                let contiguous = if !self.is_contiguous() {
                    self.contiguous()?
                } else {
                    self.clone()
                };
                let bytes = contiguous.as_raw_bytes()?;
                let data_f32: Vec<f32> = match bytemuck::try_cast_slice::<u8, f16>(bytes) {
                    Ok(slice) => slice.iter().map(|x| x.to_f32()).collect(),
                    Err(_) => {
                        let n = bytes.len() / 2;
                        (0..n)
                            .map(|i| {
                                f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]).to_f32()
                            })
                            .collect()
                    }
                };
                Ok(Tensor::new(
                    f32_vec_to_bytes(data_f32),
                    contiguous.shape_sv.clone(),
                    DType::F32,
                ))
            }
            DType::Q4_0 => {
                let bytes = self.as_raw_bytes()?;
                let n_elements = self.element_count();
                if n_elements % 32 != 0 {
                    return Err(TensorError::ConversionError(format!(
                        "Q4_0 element count {} not divisible by 32",
                        n_elements
                    )));
                }
                let n_blocks = n_elements / 32;
                let expected_bytes = n_blocks * 18;
                if bytes.len() < expected_bytes {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![expected_bytes],
                        got: vec![bytes.len()],
                    });
                }
                let mut out = vec![0.0f32; n_elements];
                for b in 0..n_blocks {
                    let block = &bytes[b * 18..(b + 1) * 18];
                    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let base = b * 32;
                    for j in 0..16 {
                        let byte = block[2 + j];
                        let lo = (byte & 0x0F) as i32 - 8;
                        let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                        out[base + j] = scale * lo as f32;
                        out[base + j + 16] = scale * hi as f32;
                    }
                }
                Ok(Tensor::new(
                    f32_vec_to_bytes(out),
                    self.shape_sv.clone(),
                    DType::F32,
                ))
            }
            DType::Q4_1 => {
                let bytes = self.as_raw_bytes()?;
                let n_elements = self.element_count();
                if n_elements % 32 != 0 {
                    return Err(TensorError::ConversionError(format!(
                        "Q4_1 element count {} not divisible by 32",
                        n_elements
                    )));
                }
                let n_blocks = n_elements / 32;
                let expected_bytes = n_blocks * 20; // Q4_1: 20 bytes/block
                if bytes.len() < expected_bytes {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![expected_bytes],
                        got: vec![bytes.len()],
                    });
                }
                let mut out = vec![0.0f32; n_elements];
                for b in 0..n_blocks {
                    let block = &bytes[b * 20..(b + 1) * 20];
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
                    let base = b * 32;
                    for j in 0..16 {
                        let byte = block[4 + j];
                        let lo = (byte & 0x0F) as f32;
                        let hi = ((byte >> 4) & 0x0F) as f32;
                        out[base + j] = d * lo + m;
                        out[base + j + 16] = d * hi + m;
                    }
                }
                Ok(Tensor::new(
                    f32_vec_to_bytes(out),
                    self.shape_sv.clone(),
                    DType::F32,
                ))
            }
            DType::Q8_0 => {
                let bytes = self.as_raw_bytes()?;
                let n_elements = self.element_count();
                if n_elements % 32 != 0 {
                    return Err(TensorError::ConversionError(format!(
                        "Q8_0 element count {} not divisible by 32",
                        n_elements
                    )));
                }
                let n_blocks = n_elements / 32;
                let expected_bytes = n_blocks * 34;
                if bytes.len() < expected_bytes {
                    return Err(TensorError::ShapeMismatch {
                        expected: vec![expected_bytes],
                        got: vec![bytes.len()],
                    });
                }
                let mut out = Vec::with_capacity(n_elements);
                for b in 0..n_blocks {
                    let block = &bytes[b * 34..(b + 1) * 34];
                    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    for j in 0..32 {
                        let quant = block[2 + j] as i8;
                        out.push(scale * quant as f32);
                    }
                }
                Ok(Tensor::new(
                    f32_vec_to_bytes(out),
                    self.shape_sv.clone(),
                    DType::F32,
                ))
            }
            _ => Err(TensorError::NotImplemented(format!(
                "Conversion from {:?} to F32",
                self.dtype
            ))),
        }
    }

    /// Convert F32 tensor to F16 for reduced memory (2 bytes/param, lossless for most weights).
    pub fn to_f16(&self) -> TensorResult<Tensor> {
        if self.dtype == DType::F16 {
            return Ok(self.clone());
        }
        let f32_tensor = self.to_f32()?;
        let f32_data = f32_tensor.data()?;
        let f16_bytes: Vec<u8> = f32_data
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        Ok(Tensor::new(f16_bytes, self.shape_sv.clone(), DType::F16))
    }

    /// Extract owned byte storage, if uniquely owned.
    pub fn into_bytes(self) -> Option<Vec<u8>> {
        match Arc::try_unwrap(self.data) {
            Ok(Storage::Owned(v)) => Some(v),
            _ => None,
        }
    }

    // ==================== Indexing ====================

    /// Get a single element by indices (as f32).
    pub fn get(&self, indices: &[usize]) -> TensorResult<f32> {
        if indices.len() != self.ndim() {
            return Err(TensorError::InvalidOperation(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }
        let mut offset = 0usize;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape_sv[i] {
                return Err(TensorError::IndexOutOfBounds {
                    dim: i,
                    index: idx,
                    size: self.shape_sv[i],
                });
            }
            offset += idx * self.strides[i];
        }
        // Read the f32 value at this offset
        let byte_offset = offset * 4; // F32 = 4 bytes
        let bytes = self.as_raw_bytes()?;
        if byte_offset + 4 > bytes.len() {
            return Err(TensorError::IndexOutOfBounds {
                dim: 0,
                index: byte_offset,
                size: bytes.len(),
            });
        }
        let val_bytes: [u8; 4] = bytes[byte_offset..byte_offset + 4].try_into().unwrap();
        Ok(f32::from_ne_bytes(val_bytes))
    }

    // ==================== Internal helpers ====================

    pub(crate) fn compute_strides_sv(shape: &[usize]) -> TensorShape {
        if shape.is_empty() {
            return SmallVec::new();
        }
        let mut strides = smallvec::smallvec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Backward-compat: compute strides for Shape type
    fn compute_strides(shape: &Shape) -> Vec<usize> {
        let sv = Self::compute_strides_sv(shape.dims());
        sv.to_vec()
    }

    pub(crate) fn normalize_dim(&self, dim: i64) -> TensorResult<usize> {
        let ndim = self.ndim() as i64;
        let normalized = if dim < 0 { dim + ndim } else { dim };
        if normalized >= 0 && normalized < ndim {
            Ok(normalized as usize)
        } else {
            Err(TensorError::InvalidDimension {
                dim,
                ndim: self.ndim(),
            })
        }
    }
}

// ==================== Tensor Iterator ====================

struct TensorIterator<'a> {
    tensor: &'a Tensor,
    indices: Vec<usize>,
    done: bool,
}

impl<'a> TensorIterator<'a> {
    fn new(tensor: &'a Tensor) -> Self {
        let indices = vec![0; tensor.ndim()];
        let done = tensor.numel() == 0;
        Self {
            tensor,
            indices,
            done,
        }
    }

    fn advance(&mut self) {
        for i in (0..self.indices.len()).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.tensor.shape_sv[i] {
                return;
            }
            self.indices[i] = 0;
        }
        self.done = true;
    }
}

impl Iterator for TensorIterator<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let val = self.tensor.get(&self.indices).ok()?;
        self.advance();
        Some(val)
    }
}

// ==================== Display ====================

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={:?}, device={:?})",
            self.shape_sv.as_slice(),
            self.dtype,
            self.device
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = Shape::new(self.shape_sv.to_vec());
        if self.numel() <= 100 {
            write!(f, "Tensor({}, {:?})", shape, self.to_vec())
        } else {
            write!(
                f,
                "Tensor({}, [{:.4}, {:.4}, ..., {:.4}, {:.4}])",
                shape,
                self.get(&vec![0; self.ndim()]).unwrap_or(0.0),
                self.iter().nth(1).unwrap_or(0.0),
                self.iter().nth(self.numel() - 2).unwrap_or(0.0),
                self.iter().last().unwrap_or(0.0),
            )
        }
    }
}

// ==================== Trait impls ====================

impl TensorOps for Tensor {
    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn dtype(&self) -> DType {
        self.dtype()
    }

    fn matmul(&self, other: &Self) -> TensorResult<Self> {
        self.matmul(other)
    }

    fn add(&self, other: &Self) -> TensorResult<Self> {
        self.add(other)
    }

    fn softmax(&self, dim: i64) -> TensorResult<Self> {
        self.softmax(dim)
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: Tensor::from_vec
    #[test]
    fn test_from_vec_correct_shape_and_dtype() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.dtype(), DType::F32);
    }

    /// @covers: Tensor::from_vec
    #[test]
    fn test_from_vec_shape_mismatch_returns_error() {
        let r = Tensor::from_vec(vec![1.0, 2.0], vec![3]);
        assert!(r.is_err(), "expected ShapeMismatch error");
    }

    /// @covers: Tensor::zeros, Tensor::ones
    #[test]
    fn test_zeros_sum_is_zero_and_ones_sum_is_numel() {
        let zeros = Tensor::zeros(vec![2, 3]);
        assert_eq!(zeros.sum_all(), 0.0);
        let ones = Tensor::ones(vec![2, 3]);
        assert_eq!(ones.sum_all(), 6.0);
    }

    /// @covers: Tensor::new
    #[test]
    fn test_new_raw_bytes_correct_shape_and_dtype() {
        let data = vec![0u8; 64];
        let t = Tensor::new(data, SmallVec::from_slice(&[4, 4]), DType::F32);
        assert_eq!(t.shape(), &[4, 4]);
        assert_eq!(t.dtype(), DType::F32);
    }

    /// @covers: Tensor::to_vec
    #[test]
    fn test_to_vec_returns_original_data() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    /// @covers: Tensor::get
    #[test]
    fn test_get_returns_correct_elements() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(t.get(&[1, 0]).unwrap(), 3.0);
        assert_eq!(t.get(&[1, 1]).unwrap(), 4.0);
    }

    /// @covers: Tensor::get
    #[test]
    fn test_get_out_of_bounds_returns_error() {
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        assert!(t.get(&[5]).is_err());
    }

    /// @covers: Tensor::get
    #[test]
    fn test_get_wrong_number_of_indices_returns_error() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert!(t.get(&[0]).is_err());
    }

    /// @covers: Tensor::full
    #[test]
    fn test_full_fills_with_given_value() {
        let t = Tensor::full(vec![3], 7.0);
        assert_eq!(t.to_vec(), vec![7.0, 7.0, 7.0]);
    }

    /// @covers: Tensor::eye
    #[test]
    fn test_eye_creates_identity_matrix() {
        let t = Tensor::eye(3);
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[0, 1]).unwrap(), 0.0);
        assert_eq!(t.get(&[1, 1]).unwrap(), 1.0);
    }

    /// @covers: Tensor::empty
    #[test]
    fn test_empty_has_zero_elements() {
        let t = Tensor::empty();
        assert_eq!(t.numel(), 0);
    }

    /// @covers: Tensor::ndim
    #[test]
    fn test_ndim_returns_dimension_count() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.ndim(), 3);
    }

    /// @covers: Tensor::is_contiguous
    #[test]
    fn test_freshly_created_tensor_is_contiguous() {
        let t = Tensor::ones(vec![2, 3]);
        assert!(t.is_contiguous());
    }

    /// @covers: Tensor::arange
    #[test]
    fn test_arange_produces_correct_sequence() {
        let t = Tensor::arange(0.0, 3.0, 1.0).unwrap();
        assert_eq!(t.to_vec(), vec![0.0, 1.0, 2.0]);
    }

    /// @covers: Tensor::arange
    #[test]
    fn test_arange_zero_step_returns_error() {
        assert!(Tensor::arange(0.0, 5.0, 0.0).is_err());
    }

    /// @covers: Tensor::as_slice_f32
    #[test]
    fn test_as_slice_f32_wrong_dtype_returns_error() {
        let t = Tensor::new(vec![0, 0], SmallVec::from_slice(&[1]), DType::F16);
        assert!(t.as_slice_f32().is_err());
    }

    /// @covers: Tensor::to_f16
    #[test]
    fn test_to_f16_changes_dtype() {
        let t = Tensor::ones(vec![4]);
        let h = t.to_f16().unwrap();
        assert_eq!(h.dtype(), DType::F16);
    }

    /// @covers: Tensor::device
    #[test]
    fn test_device_default_is_cpu() {
        let t = Tensor::zeros(vec![1]);
        assert_eq!(t.device(), crate::api::device::Device::Cpu);
    }

    /// @covers: TensorOps (trait impl)
    #[test]
    fn test_tensor_ops_trait_shape_delegates_correctly() {
        use crate::api::traits::TensorOps;
        let t = Tensor::zeros(vec![3, 4]);
        assert_eq!(TensorOps::shape(&t), &[3, 4]);
    }

    /// @covers: Tensor::to_f32
    #[test]
    fn test_to_f32_identity_for_f32_tensor() {
        let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let r = t.to_f32().unwrap();
        assert_eq!(r.dtype(), DType::F32);
        assert_eq!(r.to_vec(), vec![1.0, 2.0]);
    }

    /// @covers: Tensor::into_bytes
    #[test]
    fn test_into_bytes_returns_owned_data() {
        let t = Tensor::from_vec(vec![1.0f32], vec![1]).unwrap();
        let bytes = t.into_bytes().unwrap();
        assert_eq!(bytes.len(), 4);
    }

    /// @covers: Tensor::iter
    #[test]
    fn test_iter_yields_all_elements() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let vals: Vec<f32> = t.iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    /// @covers: Tensor::as_raw_bytes
    #[test]
    fn test_as_raw_bytes_returns_byte_slice() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]).unwrap();
        let bytes = t.as_raw_bytes().unwrap();
        assert_eq!(bytes.len(), 8); // 2 * 4 bytes
    }

    /// @covers: Tensor::as_mut_slice_f32
    #[test]
    fn test_as_mut_slice_f32_allows_mutation() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        {
            let slice = t.as_mut_slice_f32().unwrap();
            slice[0] = 99.0;
        }
        assert_eq!(t.get(&[0]).unwrap(), 99.0);
    }

    /// @covers: Tensor::element_count
    #[test]
    fn test_element_count_returns_product_of_dims() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.element_count(), 24);
    }

    /// @covers: Tensor::new_aligned
    #[test]
    fn test_new_aligned_creates_zeroed_tensor() {
        let t = Tensor::new_aligned(vec![2, 3], DType::F32);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        let data = t.as_slice_f32().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    /// @covers: Tensor::triu
    #[test]
    fn test_triu_creates_upper_triangular_matrix() {
        let t = Tensor::triu(3);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[0, 2]).unwrap(), 1.0);
        assert_eq!(t.get(&[1, 0]).unwrap(), 0.0);
    }

    /// @covers: Tensor::tril
    #[test]
    fn test_tril_creates_lower_triangular_matrix() {
        let t = Tensor::tril(3);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[0, 1]).unwrap(), 0.0);
        assert_eq!(t.get(&[2, 2]).unwrap(), 1.0);
    }

    /// @covers: Tensor::randn
    #[test]
    fn test_randn_produces_correct_shape() {
        let t = Tensor::randn(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
    }

    /// @covers: Tensor::from_mmap
    #[test]
    fn test_from_mmap_placeholder_does_not_panic() {
        // from_mmap requires an actual mmap'd file; just test the constructor path
        // by creating a temp file. This verifies the code path compiles and runs.
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("tensor_test_mmap.bin");
        let data = vec![0u8; 48]; // 12 f32s
        std::fs::write(&path, &data).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let t = Tensor::from_mmap(
            std::sync::Arc::new(mmap),
            0,
            48,
            smallvec::smallvec![3usize, 4],
            DType::F32,
        );
        assert_eq!(t.shape(), &[3, 4]);
        std::fs::remove_file(&path).ok();
    }

    /// @covers: f32_vec_to_bytes
    #[test]
    fn test_f32_vec_to_bytes_roundtrip() {
        let v = vec![1.0f32, 2.0, 3.0];
        let bytes = f32_vec_to_bytes(v.clone());
        assert_eq!(bytes.len(), 12);
        let t = Tensor::new(bytes, smallvec::smallvec![3usize], DType::F32);
        assert_eq!(t.as_slice_f32().unwrap(), &v[..]);
    }

    /// @covers: f32_slice_to_bytes
    #[test]
    fn test_f32_slice_to_bytes_length() {
        let s = [1.0f32, 2.0];
        let bytes = f32_slice_to_bytes(&s);
        assert_eq!(bytes.len(), 8);
    }
}
