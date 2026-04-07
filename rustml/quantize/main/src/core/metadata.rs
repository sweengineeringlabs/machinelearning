use rustml_gguf::GGUFValue;
use serde_json::Value;
use crate::api::error::QuantizeResult;

/// Build GGUF metadata key-value pairs from a HuggingFace config.json.
pub(crate) fn build_gguf_metadata(config_json: &Value) -> QuantizeResult<Vec<(String, GGUFValue)>> {
    let mut meta = Vec::new();

    // Determine architecture
    let model_type = config_json
        .get("model_type")
        .or_else(|| config_json.get("text_config").and_then(|tc| tc.get("model_type")))
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    let arch = normalize_arch(model_type);
    meta.push(("general.architecture".to_string(), GGUFValue::String(arch.clone())));

    // Extract text config (Gemma 4 nests it under text_config)
    let text_cfg = config_json.get("text_config").unwrap_or(config_json);

    // Required fields
    if let Some(v) = get_u32(text_cfg, "hidden_size") {
        meta.push((format!("{arch}.embedding_length"), GGUFValue::U32(v)));
    }
    if let Some(v) = get_u32(text_cfg, "num_hidden_layers") {
        meta.push((format!("{arch}.block_count"), GGUFValue::U32(v)));
    }
    if let Some(v) = get_u32(text_cfg, "num_attention_heads") {
        meta.push((format!("{arch}.attention.head_count"), GGUFValue::U32(v)));
    }
    if let Some(v) = get_u32(text_cfg, "num_key_value_heads") {
        meta.push((format!("{arch}.attention.head_count_kv"), GGUFValue::U32(v)));
    }
    if let Some(v) = get_u32(text_cfg, "intermediate_size") {
        meta.push((format!("{arch}.feed_forward_length"), GGUFValue::U32(v)));
    }
    if let Some(v) = get_u32(text_cfg, "vocab_size") {
        meta.push((format!("{arch}.vocab_size"), GGUFValue::U32(v)));
    }
    if let Some(v) = get_u32(text_cfg, "max_position_embeddings") {
        meta.push((format!("{arch}.context_length"), GGUFValue::U32(v)));
    }

    // Float fields
    if let Some(v) = get_f32(text_cfg, "rms_norm_eps") {
        meta.push((format!("{arch}.attention.layer_norm_rms_epsilon"), GGUFValue::F32(v)));
    }

    // RoPE theta (can be nested in rope_parameters)
    let rope_theta = text_cfg.get("rope_theta")
        .and_then(|v| v.as_f64())
        .or_else(|| {
            text_cfg.get("rope_parameters")
                .and_then(|rp| {
                    // Try direct or under full_attention
                    rp.get("rope_theta")
                        .or_else(|| rp.get("full_attention").and_then(|fa| fa.get("rope_theta")))
                        .and_then(|v| v.as_f64())
                })
        });
    if let Some(v) = rope_theta {
        meta.push((format!("{arch}.rope.freq_base"), GGUFValue::F32(v as f32)));
    }

    // Head dim
    if let Some(v) = get_u32(text_cfg, "head_dim") {
        meta.push((format!("{arch}.attention.key_length"), GGUFValue::U32(v)));
        meta.push((format!("{arch}.attention.value_length"), GGUFValue::U32(v)));
    }

    // Sliding window
    if let Some(v) = get_u32(text_cfg, "sliding_window") {
        meta.push((format!("{arch}.attention.sliding_window"), GGUFValue::U32(v)));
    }

    // Softcapping
    if let Some(v) = get_f32(text_cfg, "final_logit_softcapping") {
        meta.push((format!("{arch}.final_logit_softcapping"), GGUFValue::F32(v)));
    }

    // Gemma 4 specific: global head dim
    if let Some(v) = get_u32(text_cfg, "global_head_dim") {
        meta.push((format!("{arch}.attention.global_key_length"), GGUFValue::U32(v)));
    }

    // Gemma 4 specific: num_kv_shared_layers
    if let Some(v) = get_u32(text_cfg, "num_kv_shared_layers") {
        meta.push((format!("{arch}.num_kv_shared_layers"), GGUFValue::U32(v)));
    }

    // Gemma 4 specific: hidden_size_per_layer_input (PLE)
    if let Some(v) = get_u32(text_cfg, "hidden_size_per_layer_input") {
        meta.push((format!("{arch}.hidden_size_per_layer_input"), GGUFValue::U32(v)));
    }

    // Gemma 4 specific: use_double_wide_mlp
    if text_cfg.get("use_double_wide_mlp").and_then(|v| v.as_bool()) == Some(true) {
        meta.push((format!("{arch}.use_double_wide_mlp"), GGUFValue::U32(1)));
    }

    // Gemma 4 specific: layer_types array
    if let Some(layer_types) = text_cfg.get("layer_types").and_then(|v| v.as_array()) {
        let types: Vec<GGUFValue> = layer_types.iter()
            .filter_map(|v| v.as_str().map(|s| GGUFValue::String(s.to_string())))
            .collect();
        if !types.is_empty() {
            meta.push((format!("{arch}.layer_types"), GGUFValue::Array(types)));
        }
    }

    // RoPE parameters per layer type (Gemma 4)
    if let Some(rope_params) = text_cfg.get("rope_parameters").and_then(|v| v.as_object()) {
        for (layer_type, params) in rope_params {
            let prefix = format!("{arch}.rope.{}", layer_type);
            if let Some(theta) = params.get("rope_theta").and_then(|v| v.as_f64()) {
                meta.push((format!("{prefix}.freq_base"), GGUFValue::F32(theta as f32)));
            }
            if let Some(factor) = params.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
                meta.push((format!("{prefix}.partial_rotary_factor"), GGUFValue::F32(factor as f32)));
            }
            if let Some(rope_type) = params.get("rope_type").and_then(|v| v.as_str()) {
                meta.push((format!("{prefix}.type"), GGUFValue::String(rope_type.to_string())));
            }
        }
    }

    // BOS/EOS token IDs
    if let Some(v) = get_u32(config_json, "bos_token_id")
        .or_else(|| get_u32(text_cfg, "bos_token_id"))
    {
        meta.push(("tokenizer.ggml.bos_token_id".to_string(), GGUFValue::U32(v)));
    }
    if let Some(v) = config_json.get("eos_token_id").and_then(|v| {
        v.as_u64().map(|n| n as u32)
            .or_else(|| v.as_array().and_then(|arr| arr.first()).and_then(|v| v.as_u64()).map(|n| n as u32))
    }) {
        meta.push(("tokenizer.ggml.eos_token_id".to_string(), GGUFValue::U32(v)));
    }

    // General metadata
    meta.push(("general.file_type".to_string(), GGUFValue::U32(0))); // placeholder, updated by caller
    meta.push(("general.quantization_version".to_string(), GGUFValue::U32(2)));

    Ok(meta)
}

/// Normalize HuggingFace model_type to GGUF architecture string.
fn normalize_arch(model_type: &str) -> String {
    match model_type {
        "gemma4" | "gemma4_text" => "gemma2".to_string(), // Gemma 4 uses gemma2 arch in GGUF
        "gemma3" | "gemma3_text" => "gemma2".to_string(),
        "gemma2" => "gemma2".to_string(),
        "gemma" => "gemma".to_string(),
        "llama" => "llama".to_string(),
        "mistral" => "llama".to_string(),
        "qwen2" => "qwen2".to_string(),
        "phi3" | "phi" => "phi3".to_string(),
        other => other.to_string(),
    }
}

/// Extract tokenizer metadata from a HuggingFace tokenizer.json file.
///
/// Produces the GGUF tokenizer keys: tokenizer.ggml.model, tokenizer.ggml.tokens,
/// tokenizer.ggml.scores, tokenizer.ggml.token_type.
pub(crate) fn build_tokenizer_metadata(tokenizer_json: &Value) -> QuantizeResult<Vec<(String, GGUFValue)>> {
    let mut meta = Vec::new();

    // Determine tokenizer model type
    let model = tokenizer_json.get("model").ok_or_else(|| {
        crate::api::error::QuantizeError::Config("tokenizer.json missing 'model' field".into())
    })?;

    let model_type = model.get("type").and_then(|v| v.as_str()).unwrap_or("BPE");
    let ggml_model = match model_type {
        "BPE" => "gpt2",
        "Unigram" => "llama",
        _ => "gpt2",
    };
    meta.push(("tokenizer.ggml.model".to_string(), GGUFValue::String(ggml_model.to_string())));

    // Extract vocab: map of token_string -> id
    let vocab = model.get("vocab").and_then(|v| v.as_object());
    let added_tokens = tokenizer_json.get("added_tokens").and_then(|v| v.as_array());

    // Determine vocab size
    let vocab_size = vocab.map(|v| v.len()).unwrap_or(0);
    let max_id = vocab.map(|v| {
        v.values().filter_map(|id| id.as_u64()).max().unwrap_or(0) as usize
    }).unwrap_or(0);
    let total_size = if let Some(added) = added_tokens {
        let max_added = added.iter()
            .filter_map(|t| t.get("id").and_then(|v| v.as_u64()))
            .max()
            .unwrap_or(0) as usize;
        max_added.max(max_id) + 1
    } else {
        max_id + 1
    };

    // Build token arrays indexed by ID
    let mut tokens = vec![String::new(); total_size];
    let mut scores = vec![0.0f32; total_size];
    let mut token_types = vec![1i32; total_size]; // 1 = normal

    // Fill from vocab
    if let Some(vocab_map) = vocab {
        for (token, id_val) in vocab_map {
            if let Some(id) = id_val.as_u64() {
                let idx = id as usize;
                if idx < total_size {
                    tokens[idx] = token.clone();
                }
            }
        }
    }

    // Fill scores from merges (BPE merge priority = reverse index)
    if let Some(merges) = model.get("merges").and_then(|v| v.as_array()) {
        // For BPE, scores can be based on merge order (earlier = higher priority)
        // The GGUF convention for BPE uses negative merge index as score
        for (i, merge) in merges.iter().enumerate() {
            if let Some(merge_str) = merge.as_str() {
                // Merge format: "token1 token2" -> merged token
                let merged = merge_str.replace(' ', "");
                if let Some(vocab_map) = vocab {
                    if let Some(id_val) = vocab_map.get(&merged) {
                        if let Some(id) = id_val.as_u64() {
                            let idx = id as usize;
                            if idx < total_size {
                                // Higher score = merge earlier. Use negative index so
                                // earlier merges have higher (less negative) scores.
                                scores[idx] = -(i as f32);
                            }
                        }
                    }
                }
            }
        }
    }

    // Override with added_tokens (special tokens, control tokens)
    if let Some(added) = added_tokens {
        for token_obj in added {
            let id = token_obj.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let content = token_obj.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let special = token_obj.get("special").and_then(|v| v.as_bool()).unwrap_or(false);

            if id < total_size {
                tokens[id] = content.to_string();
                if special {
                    token_types[id] = 3; // control token
                }
            }
        }
    }

    // Build GGUF arrays
    let token_values: Vec<GGUFValue> = tokens.into_iter()
        .map(GGUFValue::String)
        .collect();
    let score_values: Vec<GGUFValue> = scores.into_iter()
        .map(GGUFValue::F32)
        .collect();
    let type_values: Vec<GGUFValue> = token_types.into_iter()
        .map(GGUFValue::I32)
        .collect();

    meta.push(("tokenizer.ggml.tokens".to_string(), GGUFValue::Array(token_values)));
    meta.push(("tokenizer.ggml.scores".to_string(), GGUFValue::Array(score_values)));
    meta.push(("tokenizer.ggml.token_type".to_string(), GGUFValue::Array(type_values)));

    Ok(meta)
}

fn get_u32(obj: &Value, key: &str) -> Option<u32> {
    obj.get(key).and_then(|v| v.as_u64()).map(|v| v as u32)
}

fn get_f32(obj: &Value, key: &str) -> Option<f32> {
    obj.get(key).and_then(|v| v.as_f64()).map(|v| v as f32)
}
