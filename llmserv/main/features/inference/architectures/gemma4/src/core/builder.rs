use std::collections::HashMap;
use rustml_model::{
    LlmModel, ModelBuilder, ModelConfig, ModelError, ModelResult,
    PerLayerEmbedding, TokenEmbedding, WeightMap,
};
use rustml_inference_layers::{
    FeedForward, Linear, MultiHeadAttention, NormLayer, RMSNorm, RoPEFreqs, TransformerBlock,
};
use swe_ml_tensor::{DType, Tensor};

pub struct Gemma4Builder;

impl ModelBuilder for Gemma4Builder {
    fn remap_weights(&self, weights: HashMap<String, Tensor>, config: &ModelConfig) -> HashMap<String, Tensor> {
        WeightMap::gemma4(config.n_layers).remap(weights)
    }

    fn build(&self, config: &ModelConfig, weights: HashMap<String, Tensor>) -> ModelResult<LlmModel> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let causal = config.causal;

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

        // Per-Layer Embeddings (PLE)
        let ple = if let Some(ple_dim) = config.hidden_size_per_layer_input {
            let shared_emb = TokenEmbedding::from_weights(get_tensor("ple_shared_embedding.weight")?)?;
            let model_proj = Linear::from_weights(get_weight("ple_model_projection.weight")?, None)?;
            let proj_norm = RMSNorm::from_weight_with_offset(get_tensor("ple_projection_norm.weight")?, eps, config.rms_norm_offset.unwrap_or(1.0));
            let mut gates = Vec::with_capacity(num_layers);
            let mut projections = Vec::with_capacity(num_layers);
            let mut post_norms = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                gates.push(Linear::from_weights(get_weight(&format!("layers.{i}.ple.gate.weight"))?, None)?);
                projections.push(Linear::from_weights(get_weight(&format!("layers.{i}.ple.projection.weight"))?, None)?);
                post_norms.push(RMSNorm::from_weight_with_offset(get_tensor(&format!("layers.{i}.post_ple_norm.weight"))?, eps, config.rms_norm_offset.unwrap_or(1.0)));
            }
            Some(PerLayerEmbedding::from_weights(shared_emb, ple_dim, d_model, model_proj, proj_norm, gates, projections, post_norms)?)
        } else { None };

        let offset = config.rms_norm_offset.unwrap_or(1.0);
        let layer_types = config.layer_types.as_ref();
        let rope_params_map = config.rope_parameters.as_ref();
        let default_head_dim = config.head_dim.unwrap_or(d_model / num_heads);
        let global_head_dim = config.global_head_dim.unwrap_or(default_head_dim);
        let attn_scale = Some(config.query_pre_attn_scalar.map(|s| s.sqrt()).unwrap_or(1.0));

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_type = layer_types.and_then(|lt| lt.get(i % lt.len())).map(|s| s.as_str());
            let is_global = layer_type == Some("full_attention");
            let head_dim = if is_global { global_head_dim } else { default_head_dim };

            let rope = if let (Some(lt), Some(map)) = (layer_type, rope_params_map) {
                if let Some(params) = map.get(lt) {
                    if let Some(factor) = params.partial_rotary_factor {
                        Some(RoPEFreqs::with_partial_rotation(head_dim, max_seq_len, params.rope_theta, factor))
                    } else {
                        Some(RoPEFreqs::new(head_dim, max_seq_len, params.rope_theta))
                    }
                } else {
                    Some(RoPEFreqs::new(head_dim, max_seq_len, config.rope_theta))
                }
            } else {
                let theta = if is_global { config.rope_theta } else { config.rope_local_base_freq.unwrap_or(10000.0) };
                Some(RoPEFreqs::new(head_dim, max_seq_len, theta))
            };

            let q_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.q_proj.weight"))?, None)?;
            let k_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.k_proj.weight"))?, None)?;
            let v_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.v_proj.weight"))?, None)?;
            let out_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.out_proj.weight"))?, None)?;

            let mut attention = MultiHeadAttention::from_weights_with_head_dim(
                d_model, num_heads, config.n_kv_heads, head_dim,
                q_proj, k_proj, v_proj, out_proj, causal, rope, attn_scale,
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
            if !is_global { if let Some(w) = config.sliding_window { attention.set_window_size(w); } }
            if let Some(cap) = config.attn_logit_cap { attention.set_attn_logit_cap(cap); }

            let up_proj = Linear::from_weights(get_weight(&format!("layers.{i}.feed_forward.up_proj.weight"))?, None)?;
            let gate_proj = Linear::from_weights(get_weight(&format!("layers.{i}.feed_forward.gate_proj.weight"))?, None)?;
            let down_proj = Linear::from_weights(get_weight(&format!("layers.{i}.feed_forward.down_proj.weight"))?, None)?;
            let feed_forward = FeedForward::from_weights_geglu(up_proj, gate_proj, down_proj);

            let attention_norm = RMSNorm::from_weight_with_offset(get_tensor(&format!("layers.{i}.attention_norm.weight"))?, eps, offset);
            let ffn_norm = RMSNorm::from_weight_with_offset(get_tensor(&format!("layers.{i}.ffn_norm.weight"))?, eps, offset);
            let post_attention_norm = RMSNorm::from_weight_with_offset(get_tensor(&format!("layers.{i}.post_attention_norm.weight"))?, eps, offset);
            let post_ffn_norm = RMSNorm::from_weight_with_offset(get_tensor(&format!("layers.{i}.post_ffn_norm.weight"))?, eps, offset);

            layers.push(TransformerBlock::from_weights_rms_4norm(attention, feed_forward, attention_norm, post_attention_norm, ffn_norm, post_ffn_norm));
        }

        let norm = NormLayer::RMSNorm(RMSNorm::from_weight_with_offset(get_tensor("norm.weight")?, eps, offset));
        let output = if let Ok(w) = get_weight("output.weight") { Linear::from_weights(w, None)? } else { Linear::from_weights(token_embedding.weight().clone(), None)? };

        Ok(LlmModel { token_embedding, pos_embedding: None, embd_norm: None, ple, layers, norm, output, config: config.clone() })
    }
}
