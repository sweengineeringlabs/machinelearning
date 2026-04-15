//! Int8 block-symmetric quantizer implementation.
//!
//! Produces a `QuantizedTensor` whose `data` is a 1-D U8 tensor holding the
//! raw i8 payload reinterpreted as bytes, together with a 1-D F32 `scale`
//! tensor holding one per-block scaling factor. Dequantization reverses the
//! packing exactly — no additional metadata is consulted.

use candle_core::{DType, Tensor};

use quant_api::{QuantError, QuantFormat, QuantizedTensor, Quantizer};

/// Default Int8 block-symmetric quantizer.
///
/// Contract:
/// - `format` selects the encoding. Only `QuantFormat::Int8` is implemented;
///   any other variant would surface as `QuantError::Unsupported` on use.
/// - `block_size` is the number of source elements that share a scale. Must
///   be > 0; zero is rejected at quantize time with `QuantError::Invalid`.
pub struct DefaultQuantService {
    format: QuantFormat,
    block_size: usize,
}

impl DefaultQuantService {
    /// Create a new quantizer bound to `format` and `block_size`. No
    /// validation is performed here; invalid combinations are reported when
    /// `quantize` / `dequantize` is called so the caller can surface a
    /// structured `QuantError` rather than a panic.
    pub fn new(format: QuantFormat, block_size: usize) -> Self {
        Self { format, block_size }
    }
}

impl Quantizer for DefaultQuantService {
    fn quantize(&self, tensor: &Tensor) -> Result<QuantizedTensor, QuantError> {
        match self.format {
            QuantFormat::Int8 => quantize_int8(tensor, self.block_size),
        }
    }

    fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Tensor, QuantError> {
        match quantized.format {
            QuantFormat::Int8 => dequantize_int8(quantized),
        }
    }
}

fn quantize_int8(tensor: &Tensor, block_size: usize) -> Result<QuantizedTensor, QuantError> {
    if block_size == 0 {
        return Err(QuantError::Invalid(
            "block_size must be > 0".to_string(),
        ));
    }

    let device = tensor.device().clone();
    let original_shape = tensor.shape().dims().to_vec();

    // Pull input out as a flat Vec<f32>, converting dtype if necessary.
    let as_f32 = tensor.to_dtype(DType::F32)?;
    let flat = as_f32.flatten_all()?;
    let data: Vec<f32> = flat.to_vec1::<f32>()?;

    let n = data.len();
    let n_blocks = (n + block_size - 1) / block_size;

    let mut i8_buf: Vec<i8> = Vec::with_capacity(n);
    let mut scales: Vec<f32> = Vec::with_capacity(n_blocks);

    for block_idx in 0..n_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(n);
        let block = &data[start..end];

        let mut max_abs = 0.0f32;
        for &x in block {
            let a = x.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

        for &x in block {
            let q = (x / scale).round().clamp(-127.0, 127.0) as i8;
            i8_buf.push(q);
        }
        scales.push(scale);
    }

    // Reinterpret i8 -> u8 without copying the logical values.
    let u8_bytes: &[u8] = bytemuck::cast_slice(&i8_buf);
    let data_tensor = Tensor::from_slice(u8_bytes, (n,), &device)?;
    let scale_tensor = Tensor::from_slice(scales.as_slice(), (n_blocks,), &device)?;

    Ok(QuantizedTensor {
        data: data_tensor,
        scale: scale_tensor,
        format: QuantFormat::Int8,
        block_size,
        original_shape,
    })
}

fn dequantize_int8(q: &QuantizedTensor) -> Result<Tensor, QuantError> {
    if q.format != QuantFormat::Int8 {
        return Err(QuantError::Unsupported(format!("{:?}", q.format)));
    }
    if q.block_size == 0 {
        return Err(QuantError::Invalid("block_size must be > 0".to_string()));
    }

    let expected_elems: usize = q.original_shape.iter().product();
    let actual_elems = q.data.shape().elem_count();
    if actual_elems != expected_elems {
        return Err(QuantError::Invalid(format!(
            "data length {actual_elems} != product(original_shape) {expected_elems}"
        )));
    }

    let expected_scales = (expected_elems + q.block_size - 1) / q.block_size;
    let actual_scales = q.scale.shape().elem_count();
    if actual_scales != expected_scales {
        return Err(QuantError::Invalid(format!(
            "scale length {actual_scales} != n_blocks {expected_scales}"
        )));
    }

    let device = q.data.device().clone();

    let u8_flat = q.data.flatten_all()?.to_vec1::<u8>()?;
    let i8_view: &[i8] = bytemuck::cast_slice(&u8_flat);
    let scales = q.scale.flatten_all()?.to_vec1::<f32>()?;

    let mut out = Vec::with_capacity(expected_elems);
    for (i, &qv) in i8_view.iter().enumerate() {
        let block_idx = i / q.block_size;
        let s = scales[block_idx];
        out.push(qv as f32 * s);
    }

    let tensor = Tensor::from_vec(out, q.original_shape.as_slice(), &device)?;
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // FUTURE: once a second QuantFormat variant lands, add
    // `test_int8_dequantize_rejects_format_mismatch` that constructs a
    // QuantizedTensor with the other format and asserts
    // `QuantError::Unsupported`. Not written today because forging a
    // foreign-format `QuantizedTensor` via the public API is impossible
    // while `QuantFormat::Int8` is the only variant — such a test would
    // be fake work (the match in dequantize is exhaustive on one arm).

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn test_int8_quantize_dequantize_recovers_known_signal_with_high_cosine_similarity() {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let t = Tensor::from_vec(values.clone(), (4, 64), &device).unwrap();

        let svc = DefaultQuantService::new(QuantFormat::Int8, 32);
        let q = svc.quantize(&t).unwrap();
        let d = svc.dequantize(&q).unwrap();

        assert_eq!(d.shape().dims(), &[4, 64]);
        assert_eq!(d.dtype(), DType::F32);

        let flat_d: Vec<f32> = d.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cs = cosine_similarity(&values, &flat_d);
        assert!(
            cs > 0.99,
            "cosine similarity {cs} below 0.99 — round-trip lost too much signal"
        );
    }

    #[test]
    fn test_int8_quantize_preserves_zero_tensor_exactly() {
        let device = Device::Cpu;
        let zeros = vec![0.0f32; 16];
        let t = Tensor::from_vec(zeros, (16,), &device).unwrap();

        let svc = DefaultQuantService::new(QuantFormat::Int8, 8);
        let q = svc.quantize(&t).unwrap();
        let d = svc.dequantize(&q).unwrap();

        let out: Vec<f32> = d.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(out.len(), 16);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 0.0, "index {i} not exactly 0.0: {v}");
        }
    }

    #[test]
    fn test_int8_quantize_handles_partial_trailing_block() {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..67).map(|i| (i as f32).cos()).collect();
        let t = Tensor::from_vec(values.clone(), (67,), &device).unwrap();

        let svc = DefaultQuantService::new(QuantFormat::Int8, 32);
        let q = svc.quantize(&t).unwrap();

        // Expect three blocks: 32, 32, 3.
        assert_eq!(q.scale.shape().elem_count(), 3);
        assert_eq!(q.data.shape().elem_count(), 67);
        assert_eq!(q.original_shape, vec![67]);

        let d = svc.dequantize(&q).unwrap();
        assert_eq!(d.shape().dims(), &[67]);

        let out: Vec<f32> = d.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cs = cosine_similarity(&values, &out);
        assert!(cs > 0.99, "partial-block round-trip cosine {cs} below 0.99");
    }

    #[test]
    fn test_int8_dequantize_rejects_corrupted_scale_length() {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let t = Tensor::from_vec(values, (64,), &device).unwrap();

        let svc = DefaultQuantService::new(QuantFormat::Int8, 32);
        let mut q = svc.quantize(&t).unwrap();

        // Expected scales: 2. Replace with a wrong-length scale tensor.
        let bad_scales = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0, 1.0], (5,), &device).unwrap();
        q.scale = bad_scales;

        let err = svc.dequantize(&q).unwrap_err();
        match err {
            QuantError::Invalid(msg) => {
                assert!(
                    msg.contains("scale length"),
                    "error message should mention scale length, got: {msg}"
                );
            }
            other => panic!("expected QuantError::Invalid, got {other:?}"),
        }
    }

    #[test]
    fn test_int8_quantize_rejects_zero_block_size() {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let svc = DefaultQuantService::new(QuantFormat::Int8, 0);
        let res = svc.quantize(&t);
        match res {
            Ok(_) => panic!("expected Err(QuantError::Invalid), got Ok"),
            Err(QuantError::Invalid(msg)) => assert!(msg.contains("block_size")),
            Err(other) => panic!("expected Invalid, got {other:?}"),
        }
    }
}
