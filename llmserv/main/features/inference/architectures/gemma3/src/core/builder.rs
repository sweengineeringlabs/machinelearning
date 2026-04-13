use std::collections::HashMap;
use rustml_model::{
    LlmModel, ModelBuilder, ModelConfig, ModelError, ModelResult, TokenEmbedding, WeightMap,
};
use rustml_inference_layers::{
    FeedForward, Linear, MultiHeadAttention, NormLayer, RMSNorm, RoPEFreqs, TransformerBlock,
};
use swe_ml_tensor::{DType, Tensor};

pub struct Gemma3Builder;

impl ModelBuilder for Gemma3Builder {
    fn remap_weights(&self, weights: HashMap<String, Tensor>, config: &ModelConfig) -> HashMap<String, Tensor> {
        WeightMap::gemma3(config.n_layers).remap(weights)
    }

    fn build(&self, config: &ModelConfig, weights: HashMap<String, Tensor>) -> ModelResult<LlmModel> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let causal = config.causal;
        let head_dim = config.head_dim.unwrap_or(d_model / num_heads);
        let attn_scale = config.query_pre_attn_scalar.map(|s| s.sqrt());
        let pattern = config.sliding_window_pattern.unwrap_or(6);
        let global_theta = config.rope_theta;
        let local_theta = config.rope_local_base_freq.unwrap_or(10000.0);
        let scaling = config.rope_scaling_factor.unwrap_or(1.0);

        let local_rope = RoPEFreqs::with_scaling(head_dim, max_seq_len, local_theta, scaling);
        let global_rope = RoPEFreqs::with_scaling(head_dim, max_seq_len, global_theta, scaling);

        let get_tensor = |key: &str| -> ModelResult<Tensor> {
            weights.get(key).ok_or_else(|| ModelError::Model(format!("Missing weight: {}", key))).and_then(|t| Ok(t.to_f32()?))
        };
        let get_weight = |key: &str| -> ModelResult<Tensor> {
            weights.get(key).ok_or_else(|| ModelError::Model(format!("Missing weight: {}", key))).and_then(|t| match t.dtype() {
                DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::F32 => Ok(t.clone()),
                _ => Ok(t.to_f32()?),
            })
        };

        let token_embedding = TokenEmbedding::from_weights(get_tensor("token_embedding.weight")?)?;
        let offset = config.rms_norm_offset.unwrap_or(1.0);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let is_global = (i + 1) % pattern == 0;
            let rope = if is_global { global_rope.clone() } else { local_rope.clone() };

            let q_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.q_proj.weight"))?, None)?;
            let k_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.k_proj.weight"))?, None)?;
            let v_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.v_proj.weight"))?, None)?;
            let out_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.out_proj.weight"))?, None)?;

            let mut attention = MultiHeadAttention::from_weights_with_head_dim(
                d_model, num_heads, config.n_kv_heads, head_dim,
                q_proj, k_proj, v_proj, out_proj, causal, Some(rope), attn_scale,
            )?;

            if let (Ok(qn_w), Ok(kn_w)) = (
                get_tensor(&format!("layers.{i}.attention.q_norm.weight")),
                get_tensor(&format!("layers.{i}.attention.k_norm.weight")),
            ) {
                attention.set_qk_norms(
                    RMSNorm::from_weight_with_offset(qn_w, eps, offset),
                    RMSNorm::from_weight_with_offset(kn_w, eps, offset),
                );
            }

            if !is_global {
                if let Some(w) = config.sliding_window { attention.set_window_size(w); }
            }

            let up_proj = Linear::from_weights(get_weight(&format!("layers.{i}.feed_forward.up_proj.weight"))?, None)?;
            let gate_proj = Linear::from_weights(get_weight(&format!("layers.{i}.feed_forward.gate_proj.weight"))?, None)?;
            let down_proj = Linear::from_weights(get_weight(&format!("layers.{i}.feed_forward.down_proj.weight"))?, None)?;
            let feed_forward = FeedForward::from_weights_geglu(up_proj, gate_proj, down_proj);

            let attention_norm_w = get_tensor(&format!("layers.{i}.attention_norm.weight"))?;
            let ffn_norm_w = get_tensor(&format!("layers.{i}.ffn_norm.weight"))?;
            let post_attention_norm_w = get_tensor(&format!("layers.{i}.post_attention_norm.weight"))?;
            let post_ffn_norm_w = get_tensor(&format!("layers.{i}.post_ffn_norm.weight"))?;
            if i == 0 && std::env::var("LLMSERV_DUMP_INTERMEDIATES").is_ok() {
                let stat = |name: &str, t: &Tensor| {
                    let v: Vec<f32> = t.iter().collect();
                    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
                    let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                    eprintln!("[builder L0 weight] {:28}  shape={:?}  mean={:+.4}  rms={:.4}  min={:+.4}  max={:+.4}",
                        name, t.shape(), mean, rms, mn, mx);
                };
                stat("attention_norm.weight", &attention_norm_w);
                stat("post_attention_norm.weight", &post_attention_norm_w);
                stat("ffn_norm.weight", &ffn_norm_w);
                stat("post_ffn_norm.weight", &post_ffn_norm_w);
            }
            let attention_norm = RMSNorm::from_weight_with_offset(attention_norm_w, eps, offset);
            let ffn_norm = RMSNorm::from_weight_with_offset(ffn_norm_w, eps, offset);
            let post_attention_norm = RMSNorm::from_weight_with_offset(post_attention_norm_w, eps, offset);
            let post_ffn_norm = RMSNorm::from_weight_with_offset(post_ffn_norm_w, eps, offset);

            layers.push(TransformerBlock::from_weights_rms_4norm(attention, feed_forward, attention_norm, post_attention_norm, ffn_norm, post_ffn_norm));
        }

        let norm = NormLayer::RMSNorm(RMSNorm::from_weight_with_offset(get_tensor("norm.weight")?, eps, offset));
        let output = if let Ok(w) = get_weight("output.weight") { Linear::from_weights(w, None)? } else { Linear::from_weights(token_embedding.weight().clone(), None)? };

        Ok(LlmModel { token_embedding, pos_embedding: None, embd_norm: None, ple: None, layers, norm, output, config: config.clone() })
    }
}
