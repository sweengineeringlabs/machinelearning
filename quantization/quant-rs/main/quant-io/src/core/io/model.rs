//! Model I/O — safetensors load/save and GGUF tensor loading.
//!
//! `ModelIO` is the single entry point the CLI and other crates use to
//! read model weights off disk and write quantized outputs back. All
//! errors are normalized into `quant_api::QuantError`.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use quant_api::QuantError;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

/// Maps crate-internal I/O errors to `QuantError`.
///
/// `safetensors::SafeTensorError` is not `Copy` and its variants change
/// between versions, so we stringify rather than pattern-match.
fn map_safetensors_error(e: safetensors::SafeTensorError) -> QuantError {
    QuantError::Invalid(format!("safetensors: {e}"))
}

fn map_io_error(e: std::io::Error) -> QuantError {
    QuantError::Invalid(format!("io: {e}"))
}

/// Convert a safetensors dtype into a candle dtype. Only the dtypes we
/// actually read are supported — unknown dtypes become
/// `QuantError::Unsupported` rather than a silent fallback.
fn safetensors_to_candle_dtype(dtype: Dtype) -> Result<DType, QuantError> {
    match dtype {
        Dtype::F32 => Ok(DType::F32),
        Dtype::F16 => Ok(DType::F16),
        Dtype::BF16 => Ok(DType::BF16),
        Dtype::U8 => Ok(DType::U8),
        Dtype::I64 => Ok(DType::I64),
        Dtype::U32 => Ok(DType::U32),
        Dtype::F64 => Ok(DType::F64),
        other => Err(QuantError::Unsupported(format!(
            "safetensors dtype {other:?} not supported by ModelIO::load_safetensors"
        ))),
    }
}

/// Public model I/O surface.
///
/// All methods are associated functions — `ModelIO` is a namespace, not
/// a value. The type is a zero-sized struct so it can be referenced in
/// trait bounds or moved behind a trait later without a breaking change.
pub struct ModelIO;

impl ModelIO {
    /// Load every tensor from a safetensors file into a name → Tensor
    /// map. Tensors are materialized on `Device::Cpu` using the file's
    /// native dtype (no silent f32 upcast).
    ///
    /// # Errors
    /// * `QuantError::Invalid` — the file is missing, truncated, or
    ///   malformed.
    /// * `QuantError::Unsupported` — a tensor uses a dtype that
    ///   `safetensors_to_candle_dtype` does not cover.
    /// * `QuantError::Tensor` — candle failed to construct the tensor.
    pub fn load_safetensors<P: AsRef<Path>>(
        path: P,
    ) -> Result<HashMap<String, Tensor>, QuantError> {
        let bytes = std::fs::read(path.as_ref()).map_err(map_io_error)?;
        let st = SafeTensors::deserialize(&bytes).map_err(map_safetensors_error)?;

        let mut out = HashMap::with_capacity(st.names().len());
        for (name, view) in st.tensors() {
            let candle_dtype = safetensors_to_candle_dtype(view.dtype())?;
            let shape = view.shape().to_vec();
            let tensor = Tensor::from_raw_buffer(view.data(), candle_dtype, &shape, &Device::Cpu)?;
            out.insert(name, tensor);
        }
        Ok(out)
    }

    /// Write tensors to a safetensors file. The caller owns the byte
    /// layout: each tuple is `(dtype, shape, raw little-endian bytes)`.
    /// This is deliberate — the CLI stores mixed F32 scales, F32
    /// protected weights, and U8 quantized payloads in the same file.
    ///
    /// # Errors
    /// * `QuantError::Invalid` — a tensor's byte length does not match
    ///   `dtype.size() * prod(shape)`, or the underlying writer failed.
    pub fn save_safetensors<P: AsRef<Path>>(
        path: P,
        tensors: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
    ) -> Result<(), QuantError> {
        // Build TensorViews that borrow from the HashMap's byte vectors.
        // We must keep `tensors` alive for the duration of the
        // serialize call — which we do, since `views` borrows it.
        let mut views: Vec<(String, TensorView<'_>)> = Vec::with_capacity(tensors.len());
        for (name, (dtype, shape, data)) in tensors.iter() {
            let view = TensorView::new(*dtype, shape.clone(), data.as_slice())
                .map_err(map_safetensors_error)?;
            views.push((name.clone(), view));
        }

        safetensors::tensor::serialize_to_file(views, &None, path.as_ref())
            .map_err(map_safetensors_error)?;
        Ok(())
    }

    /// Load a single named tensor from a GGUF file, dequantized to
    /// F32 on CPU.
    ///
    /// # Errors
    /// * `QuantError::Invalid` — the file is missing, malformed, or the
    ///   requested tensor name does not exist.
    /// * `QuantError::Tensor` — dequantization failed.
    pub fn load_gguf_tensor<P: AsRef<Path>>(
        path: P,
        name: &str,
    ) -> Result<Tensor, QuantError> {
        let mut file = File::open(path.as_ref()).map_err(map_io_error)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| QuantError::Invalid(format!("gguf read: {e}")))?;

        let qtensor = content
            .tensor(&mut file, name, &Device::Cpu)
            .map_err(|e| QuantError::Invalid(format!("gguf tensor '{name}': {e}")))?;

        let dequantized = qtensor.dequantize(&Device::Cpu)?;

        // Some quantized formats dequantize to F16/BF16; normalize to F32
        // so downstream evaluation code can assume a single dtype.
        if dequantized.dtype() == DType::F32 {
            Ok(dequantized)
        } else {
            Ok(dequantized.to_dtype(DType::F32)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Round-trip a small f32 matrix through save → load and verify
    /// element-wise equality. Would fail if save wrote the wrong bytes,
    /// load used the wrong dtype, or shape was mangled.
    #[test]
    fn test_save_then_load_safetensors_round_trips_f32_tensor() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = bytemuck::cast_slice::<f32, u8>(&values).to_vec();

        let mut map = HashMap::new();
        map.insert(
            "weight".to_string(),
            (Dtype::F32, vec![2usize, 2], bytes),
        );

        let tmp = NamedTempFile::new().expect("tmp file");
        let path = tmp.path().to_path_buf();
        drop(tmp); // Windows: release the handle before writing.

        ModelIO::save_safetensors(&path, map).expect("save");

        let loaded = ModelIO::load_safetensors(&path).expect("load");
        assert_eq!(loaded.len(), 1, "exactly one tensor expected");

        let t = loaded.get("weight").expect("weight present");
        assert_eq!(t.shape().dims(), &[2, 2]);
        assert_eq!(t.dtype(), DType::F32);

        let got: Vec<f32> = t
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got, values, "f32 values must round-trip exactly");
    }

    /// Round-trip a U8 payload (0..=255). Would fail if bytes were
    /// reinterpreted as a different dtype, or if padding/truncation
    /// corrupted the payload.
    #[test]
    fn test_save_then_load_safetensors_preserves_u8_payload() {
        let payload: Vec<u8> = (0u16..256).map(|v| v as u8).collect();

        let mut map = HashMap::new();
        map.insert(
            "q".to_string(),
            (Dtype::U8, vec![256usize], payload.clone()),
        );

        let tmp = NamedTempFile::new().expect("tmp");
        let path = tmp.path().to_path_buf();
        drop(tmp);

        ModelIO::save_safetensors(&path, map).expect("save");

        let loaded = ModelIO::load_safetensors(&path).expect("load");
        let t = loaded.get("q").expect("q present");
        assert_eq!(t.shape().dims(), &[256]);
        assert_eq!(t.dtype(), DType::U8);

        let got: Vec<u8> = t.to_vec1::<u8>().unwrap();
        assert_eq!(got, payload, "u8 payload must survive byte-for-byte");
    }

    /// Two named tensors in one file — would fail if save only wrote
    /// the first entry or if load silently dropped one.
    #[test]
    fn test_save_then_load_safetensors_preserves_multiple_tensors() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [10.0f32, 20.0];

        let mut map = HashMap::new();
        map.insert(
            "a".to_string(),
            (Dtype::F32, vec![3usize], bytemuck::cast_slice::<f32, u8>(&a).to_vec()),
        );
        map.insert(
            "b".to_string(),
            (Dtype::F32, vec![2usize], bytemuck::cast_slice::<f32, u8>(&b).to_vec()),
        );

        let tmp = NamedTempFile::new().expect("tmp");
        let path = tmp.path().to_path_buf();
        drop(tmp);

        ModelIO::save_safetensors(&path, map).expect("save");

        let loaded = ModelIO::load_safetensors(&path).expect("load");
        assert_eq!(loaded.len(), 2);

        let ta = loaded.get("a").expect("a");
        assert_eq!(ta.shape().dims(), &[3]);
        assert_eq!(ta.to_vec1::<f32>().unwrap(), a);

        let tb = loaded.get("b").expect("b");
        assert_eq!(tb.shape().dims(), &[2]);
        assert_eq!(tb.to_vec1::<f32>().unwrap(), b);
    }

    /// A missing file must surface as an error, not a silent empty
    /// map. Would fail if load_safetensors ever returned `Ok(HashMap::new())`.
    #[test]
    fn test_load_safetensors_returns_error_for_missing_file() {
        let err = ModelIO::load_safetensors("this/path/definitely/does/not/exist.safetensors")
            .expect_err("missing file must error");
        match err {
            QuantError::Invalid(msg) => assert!(
                msg.contains("io"),
                "expected io error, got: {msg}"
            ),
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    /// Random bytes cannot be a valid safetensors header. Would fail
    /// if the deserializer accepted garbage or panicked instead of
    /// returning an error.
    #[test]
    fn test_load_safetensors_returns_error_for_corrupted_file() {
        let mut tmp = NamedTempFile::new().expect("tmp");
        // 16 bytes: first 8 will be parsed as a bogus header length.
        tmp.write_all(&[0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 1, 2, 3, 4, 5, 6, 7, 8])
            .expect("write");
        tmp.flush().expect("flush");
        let path = tmp.path().to_path_buf();
        drop(tmp);

        let err = ModelIO::load_safetensors(&path).expect_err("corrupted file must error");
        match err {
            QuantError::Invalid(_) => {}
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    /// GGUF support is implemented, but we don't ship a fixture file
    /// in the repo — loading from a non-existent path must still error
    /// cleanly (not panic). This proves the error path works; full
    /// content correctness requires a real GGUF fixture and is covered
    /// by the CLI's `run_gguf_analysis` integration path.
    #[test]
    fn test_load_gguf_tensor_returns_error_for_missing_file() {
        let err = ModelIO::load_gguf_tensor("nope/missing.gguf", "any").expect_err("must error");
        match err {
            QuantError::Invalid(msg) => {
                assert!(msg.contains("io"), "expected io error, got: {msg}");
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    /// A truncated GGUF (just the 4-byte magic) must be rejected by
    /// the header reader. Would fail if we accepted a file without a
    /// full header.
    #[test]
    fn test_load_gguf_tensor_returns_error_for_malformed_file() {
        // Use into_temp_path() so the file lives until `path` is dropped,
        // not until `tmp` is dropped — otherwise the file vanishes before
        // load_gguf_tensor can open it and we'd be testing missing-file
        // behavior under the wrong test name.
        let mut tmp = NamedTempFile::new().expect("tmp");
        tmp.write_all(&[0x47, 0x47, 0x55, 0x46]).expect("write"); // "GGUF" only
        tmp.flush().expect("flush");
        let path = tmp.into_temp_path();

        let err =
            ModelIO::load_gguf_tensor(&path, "anything").expect_err("malformed must error");
        match err {
            QuantError::Invalid(msg) => {
                assert!(
                    msg.to_lowercase().contains("gguf"),
                    "expected gguf-origin error message, got: {msg}"
                );
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }
}
