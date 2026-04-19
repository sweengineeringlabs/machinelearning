use rustml_gguf::{GGMLType, GGUFValue};
use swe_llmmodel_download::{Download, HuggingFaceDownload};
use swe_llmmodel_io::{LoadTensors, SafeTensorsStore};

use crate::api::error::{QuantizeError, QuantizeResult};
use crate::api::traits::QuantizeEngine;
use crate::api::types::*;
use crate::core::classifier::classify_tensor;
use crate::core::metadata::build_gguf_metadata;
use crate::core::metrics::compute_metrics;
use crate::core::name_map::hf_to_gguf_name;

/// Default quantization engine implementation.
pub(crate) struct DefaultQuantizeEngine;

impl DefaultQuantizeEngine {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl QuantizeEngine for DefaultQuantizeEngine {
    fn run(&self, config: &QuantizeConfig) -> QuantizeResult<QuantizeReport> {
        run_quantize(config)
    }
}

fn run_quantize(config: &QuantizeConfig) -> QuantizeResult<QuantizeReport> {
    // Step 1: Resolve model path
    let downloader = HuggingFaceDownload::new();
    let model_dir = downloader.cache_dir().join(config.model_id.replace('/', "--"));

    let safetensors_path = model_dir.join("model.safetensors");
    let config_path = model_dir.join("config.json");

    if !safetensors_path.exists() {
        return Err(QuantizeError::ModelLoad(format!(
            "model.safetensors not found at {}. Download the model first with: \
             rustml hub download {}",
            safetensors_path.display(),
            config.model_id
        )));
    }
    if !config_path.exists() {
        return Err(QuantizeError::ModelLoad(format!(
            "config.json not found at {}",
            config_path.display()
        )));
    }

    // Step 2: Load config.json
    let config_json: serde_json::Value = {
        let data = std::fs::read_to_string(&config_path)?;
        serde_json::from_str(&data)?
    };

    let model_type = config_json
        .get("model_type")
        .or_else(|| config_json.get("text_config").and_then(|tc| tc.get("model_type")))
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_string();

    log::info!("Model type: {model_type}");

    // Step 3: Build GGUF metadata
    let mut metadata = build_gguf_metadata(&config_json)?;

    // Add tokenizer metadata if tokenizer.json exists
    let tokenizer_path = model_dir.join("tokenizer.json");
    if tokenizer_path.exists() {
        log::info!("Loading tokenizer from {}", tokenizer_path.display());
        let tok_data = std::fs::read_to_string(&tokenizer_path)?;
        let tok_json: serde_json::Value = serde_json::from_str(&tok_data)?;
        let tok_meta = crate::core::metadata::build_tokenizer_metadata(&tok_json)?;
        metadata.extend(tok_meta);
    } else {
        log::warn!("No tokenizer.json found — GGUF will lack tokenizer metadata");
    }

    // Update file_type based on target
    let file_type_value = match config.target {
        QuantTarget::Q8_0 => 7u32,  // GGUF file type for mostly Q8_0
        QuantTarget::Q4_0 => 2u32,  // mostly Q4_0
        QuantTarget::Q4_1 => 3u32,  // mostly Q4_1
    };
    for (key, val) in &mut metadata {
        if key == "general.file_type" {
            *val = GGUFValue::U32(file_type_value);
        }
    }

    // Step 4: Load safetensors via mmap
    log::info!("Loading safetensors from {}", safetensors_path.display());
    let tensors = SafeTensorsStore.load(&safetensors_path)?;

    log::info!("Loaded {} tensors", tensors.len());

    // Step 5: Quantize each tensor
    let mut writer = rustml_gguf::writer::GGUFWriter::new();

    for (key, value) in &metadata {
        writer.add_metadata(key.clone(), value.clone());
    }

    let mut report = QuantizeReport {
        total_tensors: tensors.len(),
        quantized_tensors: 0,
        skipped_tensors: 0,
        original_bytes: 0,
        quantized_bytes: 0,
        compression_ratio: 1.0,
        per_tensor: Vec::new(),
    };

    // Sort tensor names for deterministic output
    let mut tensor_names: Vec<&String> = tensors.keys().collect();
    tensor_names.sort();

    for hf_name in tensor_names {
        // Skip non-text model tensors (vision tower, audio tower, multimodal adapters)
        if is_multimodal_tensor(hf_name) {
            log::debug!("  Skipping multimodal tensor: {hf_name}");
            continue;
        }

        let tensor = &tensors[hf_name];
        let shape = tensor.shape();
        let original_bytes = tensor.as_raw_bytes().map(|b| b.len()).unwrap_or(0) as u64;
        report.original_bytes += original_bytes;

        let class = classify_tensor(hf_name);
        let gguf_name = hf_to_gguf_name(hf_name, &model_type);

        // Determine if we should quantize
        let should_quantize = class.should_quantize()
            && shape.len() >= 2
            && shape.iter().copied().max().unwrap_or(0) >= config.min_dim
            && !(config.preserve_output && class == TensorClass::Output);

        if should_quantize {
            // Convert to F32 for quantization
            let f32_tensor = tensor.to_f32()?;
            let f32_data = f32_tensor.to_vec();

            let n_elements = f32_data.len();

            // Check block alignment
            if n_elements % 32 != 0 {
                log::warn!("Skipping {hf_name}: element count {n_elements} not divisible by 32, keeping F16");
                let f16_tensor = tensor.to_f16()?;
                let data = f16_tensor.as_raw_bytes()?.to_vec();
                let qbytes = data.len() as u64;
                report.quantized_bytes += qbytes;
                report.skipped_tensors += 1;
                let dims = gguf_dims(shape);
                writer.add_tensor(gguf_name, dims, GGMLType::F16, data);
                continue;
            }

            // Quantize
            let (quantized_data, ggml_type) = quantize_f32_slice(&f32_data, config.target)?;
            let quantized_bytes = quantized_data.len() as u64;
            report.quantized_bytes += quantized_bytes;
            report.quantized_tensors += 1;

            // Compute metrics if requested
            let tensor_report = if config.show_metrics {
                let dequantized = dequantize_for_metrics(&quantized_data, n_elements, config.target)?;
                compute_metrics(
                    hf_name, &f32_data, &dequantized,
                    &format!("{:?}", tensor.dtype()), config.target.label(),
                    original_bytes, quantized_bytes,
                )
            } else {
                TensorReport {
                    name: hf_name.clone(),
                    original_dtype: format!("{:?}", tensor.dtype()),
                    target_dtype: config.target.label().to_string(),
                    original_bytes,
                    quantized_bytes,
                    mse: None,
                    max_abs_error: None,
                    snr_db: None,
                }
            };

            if config.show_metrics {
                log::info!(
                    "  {hf_name}: {} -> {} ({:.1}x) MSE={:.2e} SNR={:.1}dB",
                    format_bytes(original_bytes),
                    format_bytes(quantized_bytes),
                    original_bytes as f64 / quantized_bytes as f64,
                    tensor_report.mse.unwrap_or(0.0),
                    tensor_report.snr_db.unwrap_or(0.0),
                );
            } else {
                log::info!(
                    "  {hf_name}: {} -> {} ({:.1}x)",
                    format_bytes(original_bytes),
                    format_bytes(quantized_bytes),
                    original_bytes as f64 / quantized_bytes as f64,
                );
            }

            report.per_tensor.push(tensor_report);
            let dims = gguf_dims(shape);
            writer.add_tensor(gguf_name, dims, ggml_type, quantized_data);
        } else {
            // For norm weights: add +1.0 offset to match GGUF convention
            // (llama.cpp's converter pre-bakes the RMSNorm +1.0 shift into weights)
            let tensor_to_write = if class == TensorClass::Norm || gguf_name.contains("norm") {
                let mut f32_data = tensor.to_f32()?.to_vec();
                for v in &mut f32_data {
                    *v += 1.0;
                }
                let bytes: Vec<u8> = f32_data.iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                    .collect();
                (bytes, "F16+1.0")
            } else {
                let f16_tensor = tensor.to_f16()?;
                (f16_tensor.as_raw_bytes()?.to_vec(), "F16")
            };

            let (data, dtype_label) = tensor_to_write;
            let qbytes = data.len() as u64;
            report.quantized_bytes += qbytes;
            report.skipped_tensors += 1;

            log::info!("  {hf_name}: kept {dtype_label} ({:?}, {}) [{}]", class, format_bytes(qbytes), gguf_name);

            let dims = gguf_dims(shape);
            writer.add_tensor(gguf_name, dims, GGMLType::F16, data);
        }
    }

    // Step 5b: Handle tied embeddings — duplicate token_embd as output if needed
    let tie_embeddings = config_json
        .get("tie_word_embeddings")
        .or_else(|| config_json.get("text_config").and_then(|tc| tc.get("tie_word_embeddings")))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if tie_embeddings {
        // Check if we already have an output.weight tensor
        let has_output = tensors.keys().any(|k| {
            let mapped = hf_to_gguf_name(k, &model_type);
            mapped == "output.weight"
        });
        if !has_output {
            // Find the embedding tensor and duplicate it as output.weight
            let emb_key = tensors.keys().find(|k| {
                let mapped = hf_to_gguf_name(k, &model_type);
                mapped == "token_embd.weight"
            });
            if let Some(emb_key) = emb_key.cloned() {
                let emb_tensor = &tensors[&emb_key];
                let f16_tensor = emb_tensor.to_f16()?;
                let data = f16_tensor.as_raw_bytes()?.to_vec();
                let dims = gguf_dims(emb_tensor.shape());
                log::info!("  Tied embeddings: duplicating token_embd.weight as output.weight ({})", format_bytes(data.len() as u64));
                writer.add_tensor("output.weight", dims, GGMLType::F16, data);
            }
        }
    }

    // Step 6: Write GGUF
    log::info!("Writing GGUF to {}", config.output_path.display());
    let bytes_written = writer.write_to_file(&config.output_path)?;
    log::info!("Wrote {} GGUF file", format_bytes(bytes_written));

    report.compression_ratio = if report.quantized_bytes > 0 {
        report.original_bytes as f64 / report.quantized_bytes as f64
    } else {
        1.0
    };

    // Print summary
    log::info!("=== Quantization Summary ===");
    log::info!("  Total tensors: {}", report.total_tensors);
    log::info!("  Quantized: {}", report.quantized_tensors);
    log::info!("  Skipped (kept F32): {}", report.skipped_tensors);
    log::info!("  Original size: {}", format_bytes(report.original_bytes));
    log::info!("  Quantized size: {}", format_bytes(report.quantized_bytes));
    log::info!("  Compression ratio: {:.2}x", report.compression_ratio);

    Ok(report)
}

fn quantize_f32_slice(data: &[f32], target: QuantTarget) -> QuantizeResult<(Vec<u8>, GGMLType)> {
    match target {
        QuantTarget::Q8_0 => {
            let quantized = llmkernel::quantize_q8_0(data)?;
            Ok((quantized, GGMLType::Q8_0))
        }
        QuantTarget::Q4_0 => {
            let quantized = llmkernel::quantize_q4_0(data)?;
            Ok((quantized, GGMLType::Q4_0))
        }
        QuantTarget::Q4_1 => {
            let quantized = llmkernel::quantize_q4_1(data)?;
            Ok((quantized, GGMLType::Q4_1))
        }
    }
}

fn dequantize_for_metrics(data: &[u8], n_elements: usize, target: QuantTarget) -> QuantizeResult<Vec<f32>> {
    match target {
        QuantTarget::Q8_0 => Ok(llmkernel::dequantize_q8_0(data, n_elements)?),
        QuantTarget::Q4_0 => Ok(llmkernel::dequantize_q4_0(data, n_elements)?),
        QuantTarget::Q4_1 => Ok(llmkernel::dequantize_q4_1(data, n_elements)?),
    }
}

fn convert_to_f32_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Reverse dimensions for GGUF convention (innermost-first).
fn gguf_dims(shape: &[usize]) -> Vec<usize> {
    shape.iter().rev().copied().collect()
}

/// Check if a tensor belongs to a non-text modality (vision, audio).
fn is_multimodal_tensor(name: &str) -> bool {
    name.contains("vision_tower")
        || name.contains("audio_tower")
        || name.contains("embed_vision")
        || name.contains("embed_audio")
        || name.contains("multi_modal_projector")
        || name.contains("image_encoder")
        || name.contains("audio_encoder")
}
