//! Public configuration types for NLP models

use crate::api::error::{NlpError, NlpResult};
use rustml_core::Tensor;
pub use rustml_nn::{KVCache, PositionEncoding, PoolingStrategy};
use std::collections::HashMap;

/// Rotary parameters for a specific layer type (Gemma 4).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_type: String,
    pub rope_theta: f32,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
}

/// GPT-2 Model Configuration
#[derive(Debug, Clone)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub layer_norm_eps: f32,
}

impl GptConfig {
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn gpt2_large() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1280,
            n_layer: 36,
            n_head: 20,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn gpt2_xl() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1600,
            n_layer: 48,
            n_head: 25,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn from_hf_config(config: &serde_json::Value) -> NlpResult<Self> {
        let get_u64 = |key: &str| -> NlpResult<u64> {
            config[key].as_u64().ok_or_else(|| {
                NlpError::ModelError(format!("missing or invalid '{}' in GPT-2 config", key))
            })
        };
        Ok(Self {
            vocab_size: get_u64("vocab_size")? as usize,
            n_positions: config["n_positions"].as_u64().unwrap_or(1024) as usize,
            n_embd: get_u64("n_embd")? as usize,
            n_layer: get_u64("n_layer")? as usize,
            n_head: get_u64("n_head")? as usize,
            layer_norm_eps: config["layer_norm_epsilon"]
                .as_f64()
                .unwrap_or(1e-5) as f32,
        })
    }
}

impl Default for GptConfig {
    fn default() -> Self {
        Self::gpt2_small()
    }
}

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub do_sample: bool,
    pub repetition_penalty: f32,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 50,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            do_sample: true,
            repetition_penalty: 1.0,
            eos_token_id: Some(50256),
            pad_token_id: Some(50256),
        }
    }
}

impl GenerationConfig {
    pub fn validate(&self) -> NlpResult<()> {
        if self.temperature < 0.0 {
            return Err(NlpError::GenerationError(
                format!("temperature must be >= 0.0, got {}", self.temperature),
            ));
        }
        if let Some(k) = self.top_k {
            if k == 0 {
                return Err(NlpError::GenerationError("top_k must be > 0".into()));
            }
        }
        if let Some(p) = self.top_p {
            if p <= 0.0 || p > 1.0 {
                return Err(NlpError::GenerationError(format!("top_p must be in (0.0, 1.0], got {}", p)));
            }
        }
        if self.repetition_penalty <= 0.0 {
            return Err(NlpError::GenerationError(format!("repetition_penalty must be > 0.0, got {}", self.repetition_penalty)));
        }
        Ok(())
    }

    pub fn greedy(max_new_tokens: usize) -> Self {
        Self { max_new_tokens, do_sample: false, ..Default::default() }
    }

    pub fn with_temperature(max_new_tokens: usize, temperature: f32) -> Self {
        Self { max_new_tokens, temperature, do_sample: true, ..Default::default() }
    }

    pub fn with_top_k(max_new_tokens: usize, top_k: usize, temperature: f32) -> Self {
        Self { max_new_tokens, temperature, top_k: Some(top_k), do_sample: true, ..Default::default() }
    }

    pub fn with_top_p(max_new_tokens: usize, top_p: f32, temperature: f32) -> Self {
        Self { max_new_tokens, temperature, top_p: Some(top_p), do_sample: true, ..Default::default() }
    }
}

/// Unified model configuration supporting GPT-2, Llama, and future architectures.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: Option<usize>,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub use_bias: Option<bool>,
    #[serde(default = "default_position_encoding")]
    pub position_encoding: PositionEncoding,
    #[serde(default = "default_causal")]
    pub causal: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub attn_logit_cap: Option<f32>,
    #[serde(default)]
    pub embedding_scale: Option<f32>,
    #[serde(default)]
    pub rms_norm_offset: Option<f32>,
    #[serde(default)]
    pub attention_bias: Option<bool>,
    #[serde(default)]
    pub parallel_residual: Option<bool>,
    #[serde(default)]
    pub num_local_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub sliding_window_pattern: Option<usize>,
    #[serde(default)]
    pub query_pre_attn_scalar: Option<f32>,
    #[serde(default)]
    pub rope_local_base_freq: Option<f32>,
    #[serde(default)]
    pub rope_scaling_factor: Option<f32>,
    #[serde(default)]
    pub pooling_strategy: Option<PoolingStrategy>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    #[serde(default)]
    pub num_kv_shared_layers: Option<usize>,
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub use_double_wide_mlp: Option<bool>,
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, RopeParameters>>,
}

fn default_position_encoding() -> PositionEncoding { PositionEncoding::Learned }
fn default_causal() -> bool { true }
fn default_rope_theta() -> f32 { 10000.0 }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dim: 4096,
            hidden_dim: 11008,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: None,
            vocab_size: 32000,
            norm_eps: 1e-6,
            max_seq_len: 2048,
            use_bias: Some(false),
            position_encoding: PositionEncoding::Learned,
            causal: true,
            rope_theta: 10000.0,
            bos_token_id: None,
            eos_token_id: None,
            chat_template: None,
            sliding_window: None,
            attn_logit_cap: None,
            embedding_scale: None,
            rms_norm_offset: None,
            attention_bias: None,
            parallel_residual: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            head_dim: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            rope_local_base_freq: None,
            rope_scaling_factor: None,
            pooling_strategy: None,
            layer_types: None,
            global_head_dim: None,
            num_kv_shared_layers: None,
            hidden_size_per_layer_input: None,
            vocab_size_per_layer_input: None,
            use_double_wide_mlp: None,
            rope_parameters: None,
        }
    }
}

#[derive(serde::Deserialize)]
struct HFLlamaConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f32,
    max_position_embeddings: usize,
    rope_theta: Option<f32>,
}

#[derive(serde::Deserialize)]
struct HFMistralConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f32,
    max_position_embeddings: usize,
    rope_theta: Option<f32>,
    sliding_window: Option<usize>,
    #[serde(default)]
    attention_bias: Option<bool>,
}

#[derive(serde::Deserialize)]
struct HFGemma2Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f32,
    max_position_embeddings: usize,
    rope_theta: Option<f32>,
    #[serde(default)]
    attn_logit_softcapping: Option<f32>,
    #[serde(default)]
    sliding_window: Option<usize>,
}

#[derive(serde::Deserialize)]
struct HFFalconConfig {
    hidden_size: usize,
    n_inner: Option<usize>,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(alias = "n_head_kv")]
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    layer_norm_epsilon: Option<f32>,
    #[serde(default)]
    parallel_attn: Option<bool>,
    #[serde(default)]
    alibi: Option<bool>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
}

#[derive(serde::Deserialize)]
struct HFMixtralConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f32,
    max_position_embeddings: usize,
    rope_theta: Option<f32>,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    num_local_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
}

#[derive(serde::Deserialize)]
struct HFGemma3Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f32,
    max_position_embeddings: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    query_pre_attn_scalar: Option<f32>,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    sliding_window_pattern: Option<usize>,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_local_base_freq: Option<f32>,
    #[serde(default)]
    rope_scaling: Option<HFRopeScaling>,
}

#[derive(serde::Deserialize)]
struct HFRopeScaling {
    #[serde(default)]
    factor: Option<f32>,
}

#[derive(serde::Deserialize)]
struct HFGemma4Config {
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    vocab_size: Option<usize>,
    rms_norm_eps: Option<f32>,
    max_position_embeddings: Option<usize>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    global_head_dim: Option<usize>,
    #[serde(default)]
    num_kv_shared_layers: Option<usize>,
    #[serde(default)]
    hidden_size_per_layer_input: Option<usize>,
    #[serde(default)]
    vocab_size_per_layer_input: Option<usize>,
    #[serde(default)]
    use_double_wide_mlp: Option<bool>,
    #[serde(default)]
    rope_parameters: Option<HashMap<String, RopeParameters>>,
}

#[derive(serde::Deserialize)]
struct HFGpt2Config {
    n_embd: usize,
    n_inner: Option<usize>,
    n_layer: usize,
    n_head: usize,
    vocab_size: usize,
    n_positions: usize,
    layer_norm_epsilon: Option<f32>,
}

impl ModelConfig {
    pub fn from_hf_llama<P: AsRef<std::path::Path>>(path: P) -> NlpResult<Self> {
        let file = std::fs::File::open(&path)?;
        let hf: HFLlamaConfig = serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| NlpError::ModelError(format!("Invalid Llama config: {}", e)))?;
        Ok(Self {
            dim: hf.hidden_size, hidden_dim: hf.intermediate_size, n_layers: hf.num_hidden_layers,
            n_heads: hf.num_attention_heads, n_kv_heads: hf.num_key_value_heads, vocab_size: hf.vocab_size,
            norm_eps: hf.rms_norm_eps, max_seq_len: hf.max_position_embeddings, use_bias: Some(false),
            position_encoding: PositionEncoding::RoPE, causal: true, rope_theta: hf.rope_theta.unwrap_or(10000.0),
            bos_token_id: Some(1), eos_token_id: Some(2), ..Default::default()
        })
    }

    pub fn from_hf_gpt2<P: AsRef<std::path::Path>>(path: P) -> NlpResult<Self> {
        let file = std::fs::File::open(&path)?;
        let hf: HFGpt2Config = serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| NlpError::ModelError(format!("Invalid GPT-2 config: {}", e)))?;
        Ok(Self {
            dim: hf.n_embd, hidden_dim: hf.n_inner.unwrap_or(4 * hf.n_embd), n_layers: hf.n_layer,
            n_heads: hf.n_head, vocab_size: hf.vocab_size, norm_eps: hf.layer_norm_epsilon.unwrap_or(1e-5),
            max_seq_len: hf.n_positions, use_bias: Some(true), position_encoding: PositionEncoding::Learned,
            causal: true, eos_token_id: Some(50256), ..Default::default()
        })
    }

    pub fn from_json_value(config: &serde_json::Value) -> NlpResult<Self> {
        let model_type = config["model_type"].as_str().unwrap_or("unknown");
        println!("DEBUG: model_type='{}' keys: {:?}", model_type, config.as_object().map(|o| o.keys().collect::<Vec<_>>()));
        match model_type {
            "llama" => {
                let hf: HFLlamaConfig = serde_json::from_value(config.clone())
                    .map_err(|e| NlpError::ModelError(format!("Invalid Llama config: {}", e)))?;
                Ok(Self {
                    dim: hf.hidden_size, hidden_dim: hf.intermediate_size, n_layers: hf.num_hidden_layers,
                    n_heads: hf.num_attention_heads, n_kv_heads: hf.num_key_value_heads, vocab_size: hf.vocab_size,
                    norm_eps: hf.rms_norm_eps, max_seq_len: hf.max_position_embeddings, use_bias: Some(false),
                    position_encoding: PositionEncoding::RoPE, causal: true, rope_theta: hf.rope_theta.unwrap_or(10000.0),
                    bos_token_id: Some(1), eos_token_id: Some(2), ..Default::default()
                })
            }
            "mistral" | "qwen2" | "phi3" => {
                let hf: HFMistralConfig = serde_json::from_value(config.clone())
                    .map_err(|e| NlpError::ModelError(format!("Invalid config: {}", e)))?;
                Ok(Self {
                    dim: hf.hidden_size, hidden_dim: hf.intermediate_size, n_layers: hf.num_hidden_layers,
                    n_heads: hf.num_attention_heads, n_kv_heads: hf.num_key_value_heads, vocab_size: hf.vocab_size,
                    norm_eps: hf.rms_norm_eps, max_seq_len: hf.max_position_embeddings, position_encoding: PositionEncoding::RoPE,
                    causal: true, rope_theta: hf.rope_theta.unwrap_or(10000.0), sliding_window: hf.sliding_window,
                    attention_bias: hf.attention_bias, bos_token_id: Some(1), eos_token_id: Some(2), ..Default::default()
                })
            }
            "gemma2" => {
                let hf: HFGemma2Config = serde_json::from_value(config.clone())
                    .map_err(|e| NlpError::ModelError(format!("Invalid config: {}", e)))?;
                Ok(Self {
                    dim: hf.hidden_size, hidden_dim: hf.intermediate_size, n_layers: hf.num_hidden_layers,
                    n_heads: hf.num_attention_heads, n_kv_heads: hf.num_key_value_heads, vocab_size: hf.vocab_size,
                    norm_eps: hf.rms_norm_eps, max_seq_len: hf.max_position_embeddings, position_encoding: PositionEncoding::RoPE,
                    causal: true, rope_theta: hf.rope_theta.unwrap_or(10000.0), sliding_window: hf.sliding_window,
                    attn_logit_cap: hf.attn_logit_softcapping, embedding_scale: Some((hf.hidden_size as f32).sqrt()),
                    rms_norm_offset: Some(1.0), bos_token_id: Some(1), eos_token_id: Some(2), ..Default::default()
                })
            }
            "gemma3" | "gemma3_text" => {
                let hf: HFGemma3Config = serde_json::from_value(config.clone())
                    .map_err(|e| NlpError::ModelError(format!("Invalid config: {}", e)))?;
                let scaling_factor = hf.rope_scaling.as_ref().and_then(|s| s.factor);
                Ok(Self {
                    dim: hf.hidden_size, hidden_dim: hf.intermediate_size, n_layers: hf.num_hidden_layers,
                    n_heads: hf.num_attention_heads, n_kv_heads: hf.num_key_value_heads, vocab_size: hf.vocab_size,
                    norm_eps: hf.rms_norm_eps, max_seq_len: hf.max_position_embeddings, position_encoding: PositionEncoding::RoPE,
                    causal: true, rope_theta: hf.rope_theta.unwrap_or(1000000.0), head_dim: hf.head_dim,
                    sliding_window_pattern: hf.sliding_window_pattern, query_pre_attn_scalar: hf.query_pre_attn_scalar,
                    rope_local_base_freq: hf.rope_local_base_freq, rope_scaling_factor: scaling_factor,
                    embedding_scale: Some((hf.hidden_size as f32).sqrt()), rms_norm_offset: Some(1.0),
                    bos_token_id: Some(2), eos_token_id: Some(1), ..Default::default()
                })
            }
            "gemma4" | "gemma4_text" => {
                let mut merged = config.clone();
                if let Some(text_obj) = config.get("text_config").and_then(|v| v.as_object()) {
                    if let Some(merged_obj) = merged.as_object_mut() {
                        for (k, v) in text_obj {
                            merged_obj.insert(k.clone(), v.clone());
                        }
                    }
                }
                let hf: HFGemma4Config = serde_json::from_value(merged)
                    .map_err(|e| NlpError::ModelError(format!("Invalid config: {}", e)))?;
                
                let dim = hf.hidden_size.ok_or_else(|| NlpError::ModelError("missing field hidden_size".into()))?;
                let hidden_dim = hf.intermediate_size.ok_or_else(|| NlpError::ModelError("missing field intermediate_size".into()))?;
                let n_layers = hf.num_hidden_layers.ok_or_else(|| NlpError::ModelError("missing field num_hidden_layers".into()))?;
                let n_heads = hf.num_attention_heads.ok_or_else(|| NlpError::ModelError("missing field num_attention_heads".into()))?;
                let vocab_size = hf.vocab_size.ok_or_else(|| NlpError::ModelError("missing field vocab_size".into()))?;

                Ok(Self {
                    dim,
                    hidden_dim,
                    n_layers,
                    n_heads,
                    n_kv_heads: hf.num_key_value_heads,
                    vocab_size,
                    norm_eps: hf.rms_norm_eps.unwrap_or(1e-6),
                    max_seq_len: hf.max_position_embeddings.unwrap_or(2048),
                    position_encoding: PositionEncoding::RoPE,
                    causal: true,
                    rope_theta: 1000000.0,
                    layer_types: hf.layer_types,
                    global_head_dim: hf.global_head_dim,
                    num_kv_shared_layers: hf.num_kv_shared_layers,
                    hidden_size_per_layer_input: hf.hidden_size_per_layer_input,
                    vocab_size_per_layer_input: hf.vocab_size_per_layer_input,
                    use_double_wide_mlp: hf.use_double_wide_mlp,
                    rope_parameters: hf.rope_parameters,
                    embedding_scale: Some((dim as f32).sqrt()),
                    rms_norm_offset: Some(1.0),
                    bos_token_id: Some(2),
                    eos_token_id: Some(1),
                    ..Default::default()
                })
            }
            "falcon" => {
                let hf: HFFalconConfig = serde_json::from_value(config.clone())
                    .map_err(|e| NlpError::ModelError(format!("Invalid config: {}", e)))?;
                let use_alibi = hf.alibi.unwrap_or(false);
                Ok(Self {
                    dim: hf.hidden_size, hidden_dim: hf.n_inner.unwrap_or(4 * hf.hidden_size), n_layers: hf.num_hidden_layers,
                    n_heads: hf.num_attention_heads, n_kv_heads: hf.num_key_value_heads, vocab_size: hf.vocab_size,
                    norm_eps: hf.layer_norm_epsilon.unwrap_or(1e-5), max_seq_len: hf.max_position_embeddings.unwrap_or(2048),
                    use_bias: Some(true), position_encoding: if use_alibi { PositionEncoding::ALiBi } else { PositionEncoding::RoPE },
                    causal: true, rope_theta: 10000.0, parallel_residual: Some(hf.parallel_attn.unwrap_or(true)),
                    bos_token_id: Some(1), eos_token_id: Some(11), ..Default::default()
                })
            }
            "mixtral" => {
                let hf: HFMixtralConfig = serde_json::from_value(config.clone())
                    .map_err(|e| NlpError::ModelError(format!("Invalid config: {}", e)))?;
                Ok(Self {
                    dim: hf.hidden_size, hidden_dim: hf.intermediate_size, n_layers: hf.num_hidden_layers,
                    n_heads: hf.num_attention_heads, n_kv_heads: hf.num_key_value_heads, vocab_size: hf.vocab_size,
                    norm_eps: hf.rms_norm_eps, max_seq_len: hf.max_position_embeddings, position_encoding: PositionEncoding::RoPE,
                    causal: true, rope_theta: hf.rope_theta.unwrap_or(10000.0), sliding_window: hf.sliding_window,
                    num_local_experts: hf.num_local_experts, num_experts_per_tok: hf.num_experts_per_tok,
                    bos_token_id: Some(1), eos_token_id: Some(2), ..Default::default()
                })
            }
            _ => {
                let hf: HFGpt2Config = serde_json::from_value(config.clone())
                    .map_err(|e| NlpError::ModelError(format!("Invalid config: {}", e)))?;
                Ok(Self {
                    dim: hf.n_embd, hidden_dim: hf.n_inner.unwrap_or(4 * hf.n_embd), n_layers: hf.n_layer,
                    n_heads: hf.n_head, vocab_size: hf.vocab_size, norm_eps: hf.layer_norm_epsilon.unwrap_or(1e-5),
                    max_seq_len: hf.n_positions, use_bias: Some(true), position_encoding: PositionEncoding::Learned,
                    causal: true, eos_token_id: Some(50256), ..Default::default()
                })
            }
        }
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> NlpResult<Self> {
        let file = std::fs::File::open(&path)?;
        let config: ModelConfig = serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| NlpError::ModelError(format!("Invalid config: {}", e)))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> NlpResult<()> {
        if self.dim == 0 || self.n_heads == 0 || self.vocab_size == 0 || self.n_layers == 0 || self.max_seq_len == 0 {
            return Err(NlpError::ModelError("config fields must be > 0".into()));
        }
        if self.dim % self.n_heads != 0 {
            return Err(NlpError::ModelError(format!("dim ({}) must be divisible by n_heads ({})", self.dim, self.n_heads)));
        }
        if let Some(n_kv) = self.n_kv_heads {
            if n_kv == 0 || self.n_heads % n_kv != 0 {
                return Err(NlpError::ModelError("invalid n_kv_heads".into()));
            }
        }
        Ok(())
    }
}

pub trait LanguageModel {
    fn forward(&self, input_ids: &Tensor) -> NlpResult<Tensor>;
    fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KVCache) -> NlpResult<Tensor>;
    fn vocab_size(&self) -> usize;
    fn max_sequence_length(&self) -> usize;
    fn embedding_dim(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_config_from_json() {
        let json = serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "sliding_window": 4096
        });
        let c = ModelConfig::from_json_value(&json).unwrap();
        assert_eq!(c.dim, 4096);
        assert_eq!(c.n_kv_heads, Some(8));
        assert_eq!(c.sliding_window, Some(4096));
    }

    #[test]
    fn test_gemma3_4b_config_from_json() {
        let json = serde_json::json!({
            "model_type": "gemma3_text",
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_hidden_layers": 34,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "query_pre_attn_scalar": 256,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
            "rms_norm_eps": 1e-6,
            "vocab_size": 262208,
            "max_position_embeddings": 32768
        });
        let c = ModelConfig::from_json_value(&json).unwrap();
        assert_eq!(c.dim, 2560);
        assert_eq!(c.rope_scaling_factor, Some(8.0));
    }

    #[test]
    fn test_gemma4_config_from_json() {
        let json = serde_json::json!({
            "model_type": "gemma4_text",
            "hidden_size": 2048,
            "intermediate_size": 16384,
            "num_hidden_layers": 35,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 256000,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 8192,
            "layer_types": ["sliding_attention", "full_attention"],
            "global_head_dim": 256,
            "num_kv_shared_layers": 15,
            "hidden_size_per_layer_input": 512,
            "vocab_size_per_layer_input": 128,
            "use_double_wide_mlp": true,
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25
                }
            }
        });
        let c = ModelConfig::from_json_value(&json).unwrap();
        assert_eq!(c.dim, 2048);
        assert_eq!(c.num_kv_shared_layers, Some(15));
        let rope_params = c.rope_parameters.as_ref().unwrap();
        assert_eq!(rope_params.get("full_attention").unwrap().partial_rotary_factor, Some(0.25));
    }
}
