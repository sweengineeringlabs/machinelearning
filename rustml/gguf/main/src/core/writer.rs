use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::api::error::{GgufError, GgufResult};
use crate::api::types::{GGMLType, GGUFValue};

/// GGUF v3 binary file writer.
pub struct GGUFWriter {
    metadata: Vec<(String, GGUFValue)>,
    tensors: Vec<WriteTensor>,
}

struct WriteTensor {
    name: String,
    dimensions: Vec<usize>,
    ggml_type: GGMLType,
    data: Vec<u8>,
}

/// Align `pos` up to the next multiple of 32.
fn align32(pos: u64) -> u64 {
    (pos + 31) & !31
}

impl GGUFWriter {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn add_metadata(&mut self, key: impl Into<String>, value: GGUFValue) {
        self.metadata.push((key.into(), value));
    }

    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dimensions: Vec<usize>,
        ggml_type: GGMLType,
        data: Vec<u8>,
    ) {
        self.tensors.push(WriteTensor {
            name: name.into(),
            dimensions,
            ggml_type,
            data,
        });
    }

    pub fn write_to_file(&self, path: &Path) -> GgufResult<u64> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        let mut pos: u64 = 0;

        // Magic
        w.write_all(&[0x47, 0x47, 0x55, 0x46])?;
        pos += 4;

        // Version
        w.write_all(&3u32.to_le_bytes())?;
        pos += 4;

        // Tensor count
        w.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
        pos += 8;

        // Metadata KV count
        w.write_all(&(self.metadata.len() as u64).to_le_bytes())?;
        pos += 8;

        // Metadata entries
        for (key, value) in &self.metadata {
            pos += write_string(&mut w, key)?;
            pos += write_value(&mut w, value)?;
        }

        // Tensor info entries — we need to compute offsets relative to data section start.
        // First, figure out the header size after all tensor infos are written so we
        // know where the data section begins.
        let mut tensor_info_size: u64 = 0;
        for t in &self.tensors {
            // string name
            tensor_info_size += 8 + t.name.len() as u64;
            // n_dims(u32) + dims(u64 each) + ggml_type(u32) + offset(u64)
            tensor_info_size += 4 + (t.dimensions.len() as u64) * 8 + 4 + 8;
        }

        let data_section_start = align32(pos + tensor_info_size);

        // Compute per-tensor offsets relative to data section start.
        let mut tensor_offsets: Vec<u64> = Vec::with_capacity(self.tensors.len());
        let mut current_offset: u64 = 0;
        for t in &self.tensors {
            // Each tensor is 32-byte aligned within the data section.
            current_offset = align32(current_offset);
            tensor_offsets.push(current_offset);
            current_offset += t.data.len() as u64;
        }

        // Write tensor infos
        for (i, t) in self.tensors.iter().enumerate() {
            pos += write_string(&mut w, &t.name)?;

            w.write_all(&(t.dimensions.len() as u32).to_le_bytes())?;
            pos += 4;

            for &dim in &t.dimensions {
                w.write_all(&(dim as u64).to_le_bytes())?;
                pos += 8;
            }

            w.write_all(&t.ggml_type.to_u32().to_le_bytes())?;
            pos += 4;

            w.write_all(&tensor_offsets[i].to_le_bytes())?;
            pos += 8;
        }

        // Padding to 32-byte alignment for data section
        let padded = align32(pos);
        if padded > pos {
            let pad = vec![0u8; (padded - pos) as usize];
            w.write_all(&pad)?;
            pos = padded;
        }

        debug_assert_eq!(pos, data_section_start);

        // Write tensor data, each 32-byte aligned
        for t in &self.tensors {
            let aligned = align32(pos);
            if aligned > pos {
                let pad = vec![0u8; (aligned - pos) as usize];
                w.write_all(&pad)?;
                pos = aligned;
            }
            w.write_all(&t.data)?;
            pos += t.data.len() as u64;
        }

        w.flush()?;
        Ok(pos)
    }
}

/// Write a GGUF string (u64 length + utf8 bytes). Returns bytes written.
fn write_string(w: &mut impl Write, s: &str) -> Result<u64, GgufError> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(8 + bytes.len() as u64)
}

/// Write the type_id (u32) followed by the value payload. Returns bytes written.
fn write_value(w: &mut impl Write, value: &GGUFValue) -> Result<u64, GgufError> {
    match value {
        GGUFValue::U8(v) => {
            w.write_all(&0u32.to_le_bytes())?;
            w.write_all(&[*v])?;
            Ok(5)
        }
        GGUFValue::I8(v) => {
            w.write_all(&1u32.to_le_bytes())?;
            w.write_all(&[*v as u8])?;
            Ok(5)
        }
        GGUFValue::U16(v) => {
            w.write_all(&2u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            Ok(6)
        }
        GGUFValue::I16(v) => {
            w.write_all(&3u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            Ok(6)
        }
        GGUFValue::U32(v) => {
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            Ok(8)
        }
        GGUFValue::I32(v) => {
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            Ok(8)
        }
        GGUFValue::F32(v) => {
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            Ok(8)
        }
        GGUFValue::Bool(v) => {
            w.write_all(&7u32.to_le_bytes())?;
            w.write_all(&[if *v { 1u8 } else { 0u8 }])?;
            Ok(5)
        }
        GGUFValue::String(s) => {
            w.write_all(&8u32.to_le_bytes())?;
            let n = write_string(w, s)?;
            Ok(4 + n)
        }
        GGUFValue::Array(items) => {
            w.write_all(&9u32.to_le_bytes())?;
            let elem_type = infer_array_element_type(items);
            w.write_all(&elem_type.to_le_bytes())?;
            w.write_all(&(items.len() as u64).to_le_bytes())?;
            let mut total: u64 = 4 + 4 + 8; // type_id + elem_type + count
            for item in items {
                total += write_value_payload(w, item)?;
            }
            Ok(total)
        }
        GGUFValue::U64(v) => {
            w.write_all(&10u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            Ok(12)
        }
        GGUFValue::I64(v) => {
            w.write_all(&11u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            Ok(12)
        }
        GGUFValue::F64(v) => {
            w.write_all(&12u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
            Ok(12)
        }
    }
}

/// Write just the payload of a value (no type_id prefix). Used for array elements.
fn write_value_payload(w: &mut impl Write, value: &GGUFValue) -> Result<u64, GgufError> {
    match value {
        GGUFValue::U8(v) => { w.write_all(&[*v])?; Ok(1) }
        GGUFValue::I8(v) => { w.write_all(&[*v as u8])?; Ok(1) }
        GGUFValue::U16(v) => { w.write_all(&v.to_le_bytes())?; Ok(2) }
        GGUFValue::I16(v) => { w.write_all(&v.to_le_bytes())?; Ok(2) }
        GGUFValue::U32(v) => { w.write_all(&v.to_le_bytes())?; Ok(4) }
        GGUFValue::I32(v) => { w.write_all(&v.to_le_bytes())?; Ok(4) }
        GGUFValue::F32(v) => { w.write_all(&v.to_le_bytes())?; Ok(4) }
        GGUFValue::Bool(v) => { w.write_all(&[if *v { 1 } else { 0 }])?; Ok(1) }
        GGUFValue::String(s) => write_string(w, s),
        GGUFValue::U64(v) => { w.write_all(&v.to_le_bytes())?; Ok(8) }
        GGUFValue::I64(v) => { w.write_all(&v.to_le_bytes())?; Ok(8) }
        GGUFValue::F64(v) => { w.write_all(&v.to_le_bytes())?; Ok(8) }
        GGUFValue::Array(_) => Err(GgufError::InvalidFormat(
            "Nested arrays are not supported in GGUF".to_string(),
        )),
    }
}

/// Infer the GGUF type_id for array elements from the first element.
/// Returns 4 (u32) as a fallback for empty arrays.
fn infer_array_element_type(items: &[GGUFValue]) -> u32 {
    match items.first() {
        Some(GGUFValue::U8(_)) => 0,
        Some(GGUFValue::I8(_)) => 1,
        Some(GGUFValue::U16(_)) => 2,
        Some(GGUFValue::I16(_)) => 3,
        Some(GGUFValue::U32(_)) => 4,
        Some(GGUFValue::I32(_)) => 5,
        Some(GGUFValue::F32(_)) => 6,
        Some(GGUFValue::Bool(_)) => 7,
        Some(GGUFValue::String(_)) => 8,
        Some(GGUFValue::Array(_)) => 9,
        Some(GGUFValue::U64(_)) => 10,
        Some(GGUFValue::I64(_)) => 11,
        Some(GGUFValue::F64(_)) => 12,
        None => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::GGUFFile;

    #[test]
    fn test_write_and_read_roundtrip_metadata_and_tensor() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_gguf_writer_roundtrip.gguf");

        // Build a GGUF file with metadata and one tensor
        let mut writer = GGUFWriter::new();
        writer.add_metadata("general.architecture", GGUFValue::String("test".to_string()));
        writer.add_metadata("test.layer_count", GGUFValue::U32(42));

        // 2x3 F32 tensor = 24 bytes
        let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let raw_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer.add_tensor("blk.0.weight", vec![3, 2], GGMLType::F32, raw_bytes.clone());

        let bytes_written = writer.write_to_file(&path).expect("write_to_file failed");
        assert!(bytes_written > 0, "Should write more than 0 bytes");

        // Read it back with the existing parser
        let parsed = GGUFFile::parse_header(&path).expect("parse_header failed");

        // Verify version
        assert_eq!(parsed.version, 3);

        // Verify metadata
        assert_eq!(
            parsed.metadata.get("general.architecture")
                .and_then(|v| v.as_string())
                .unwrap(),
            "test"
        );
        assert_eq!(
            parsed.metadata.get("test.layer_count")
                .and_then(|v| v.as_u32())
                .unwrap(),
            42
        );

        // Verify tensor info
        assert_eq!(parsed.tensor_infos.len(), 1);
        let ti = &parsed.tensor_infos[0];
        assert_eq!(ti.name, "blk.0.weight");
        assert_eq!(ti.dimensions, vec![3, 2]);
        assert_eq!(ti.ggml_type, GGMLType::F32);

        // Verify data offset is 32-byte aligned
        assert_eq!(parsed.data_offset % 32, 0);

        // Read raw file bytes and verify the tensor data at the computed offset
        let file_bytes = std::fs::read(&path).unwrap();
        let tensor_start = parsed.data_offset + ti.offset as usize;
        let tensor_end = tensor_start + raw_bytes.len();
        assert_eq!(&file_bytes[tensor_start..tensor_end], raw_bytes.as_slice());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_empty_file_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_gguf_writer_empty.gguf");

        let writer = GGUFWriter::new();
        writer.write_to_file(&path).expect("write_to_file failed");

        let parsed = GGUFFile::parse_header(&path).expect("parse_header failed");
        assert_eq!(parsed.version, 3);
        assert!(parsed.metadata.is_empty());
        assert!(parsed.tensor_infos.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_multiple_tensors_alignment() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_gguf_writer_multi_tensor.gguf");

        let mut writer = GGUFWriter::new();

        // First tensor: 3 floats = 12 bytes (not 32-aligned)
        let data1: Vec<u8> = vec![1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        writer.add_tensor("t1", vec![3], GGMLType::F32, data1);

        // Second tensor: 2 floats = 8 bytes
        let data2: Vec<u8> = vec![4.0f32, 5.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        writer.add_tensor("t2", vec![2], GGMLType::F32, data2);

        writer.write_to_file(&path).expect("write_to_file failed");

        let parsed = GGUFFile::parse_header(&path).expect("parse_header failed");
        assert_eq!(parsed.tensor_infos.len(), 2);

        // Second tensor offset must be 32-byte aligned
        let t2_offset = parsed.tensor_infos[1].offset;
        assert_eq!(t2_offset % 32, 0, "Second tensor offset {t2_offset} not 32-byte aligned");

        let _ = std::fs::remove_file(&path);
    }
}
