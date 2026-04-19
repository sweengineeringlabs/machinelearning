use std::collections::HashMap;
use swe_llmmodel_model::{
    LlmModel, ModelBuilder, ModelConfig, ModelError, ModelResult,
    TokenEmbedding, WeightMap,
};
use swe_llmmodel_layers::{
    Activation, FeedForward, Linear, MultiHeadAttention, NormLayer, PositionEncoding,
    RMSNorm, TransformerBlock,
};
use swe_ml_tensor::{DType, Tensor};

/// Llama/Qwen/Mistral/Phi architecture builder.
pub struct LlamaBuilder;

impl ModelBuilder for LlamaBuilder {
    fn remap_weights(
        &self,
        weights: HashMap<String, Tensor>,
        config: &ModelConfig,
    ) -> HashMap<String, Tensor> {
        let wm = if config.attention_bias == Some(true) {
            WeightMap::llama2_with_attn_bias(config.n_layers)
        } else {
            WeightMap::llama2(config.n_layers)
        };
        wm.remap(weights)
    }

    fn build(
        &self,
        config: &ModelConfig,
        weights: HashMap<String, Tensor>,
    ) -> ModelResult<LlmModel> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let position_encoding = config.position_encoding;
        let causal = config.causal;
        let rope_theta = config.rope_theta;

        let get_tensor = |key: &str| -> ModelResult<Tensor> {
            weights
                .get(key)
                .ok_or_else(|| ModelError::Model(format!("Missing weight: {}", key)))
                .and_then(|t| Ok(t.to_f32()?))
        };
        let get_weight = |key: &str| -> ModelResult<Tensor> {
            weights
                .get(key)
                .ok_or_else(|| ModelError::Model(format!("Missing weight: {}", key)))
                .and_then(|t| match t.dtype() {
                    DType::Q4_0 | DType::Q4_1 | DType::Q8_0 | DType::F32 => Ok(t.clone()),
                    _ => Ok(t.to_f32()?),
                })
        };

        let token_embedding = TokenEmbedding::from_weights(get_tensor("token_embedding.weight")?)?;

        let pos_embedding = if position_encoding == PositionEncoding::Learned {
            if let Ok(pos_weight) = get_tensor("pos_embedding.weight") {
                Some(TokenEmbedding::from_weights(pos_weight)?)
            } else {
                Some(TokenEmbedding::new(max_seq_len, d_model))
            }
        } else {
            None
        };

        let has_attn_bias = config.attention_bias == Some(true);
        let get_bias_opt =
            |key: &str| -> Option<Tensor> { weights.get(key).and_then(|t| t.to_f32().ok()) };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let q_bias = if has_attn_bias { get_bias_opt(&format!("layers.{}.attention.q_proj.bias", i)) } else { None };
            let k_bias = if has_attn_bias { get_bias_opt(&format!("layers.{}.attention.k_proj.bias", i)) } else { None };
            let v_bias = if has_attn_bias { get_bias_opt(&format!("layers.{}.attention.v_proj.bias", i)) } else { None };
            let o_bias = if has_attn_bias { get_bias_opt(&format!("layers.{}.attention.out_proj.bias", i)) } else { None };

            let q_proj = Linear::from_weights(get_weight(&format!("layers.{}.attention.q_proj.weight", i))?, q_bias)?;
            let k_proj = Linear::from_weights(get_weight(&format!("layers.{}.attention.k_proj.weight", i))?, k_bias)?;
            let v_proj = Linear::from_weights(get_weight(&format!("layers.{}.attention.v_proj.weight", i))?, v_bias)?;
            let out_proj = Linear::from_weights(get_weight(&format!("layers.{}.attention.out_proj.weight", i))?, o_bias)?;

            let mut attention = MultiHeadAttention::from_weights(
                d_model, num_heads, config.n_kv_heads,
                q_proj, k_proj, v_proj, out_proj,
                causal, position_encoding, max_seq_len, rope_theta,
            )?;

            if let Some(window) = config.sliding_window {
                let apply = if config.attn_logit_cap.is_some() { i % 2 == 0 } else { true };
                if apply { attention.set_window_size(window); }
            }
            if let Some(cap) = config.attn_logit_cap {
                attention.set_attn_logit_cap(cap);
            }

            let up_proj = Linear::from_weights(get_weight(&format!("layers.{}.feed_forward.up_proj.weight", i))?, None)?;
            let down_proj = Linear::from_weights(get_weight(&format!("layers.{}.feed_forward.down_proj.weight", i))?, None)?;

            let gate_key = format!("layers.{}.feed_forward.gate_proj.weight", i);
            let feed_forward = if let Ok(gate_weight) = get_weight(&gate_key) {
                let gate_proj = Linear::from_weights(gate_weight, None)?;
                FeedForward::from_weights_swiglu(up_proj, gate_proj, down_proj)
            } else {
                FeedForward::from_weights_with_activation(up_proj, down_proj, Activation::Silu)
            };

            let offset = config.rms_norm_offset.unwrap_or(0.0);
            let attention_norm = RMSNorm::from_weight_with_offset(
                get_tensor(&format!("layers.{}.attention_norm.weight", i))?, eps, offset,
            );
            let ffn_norm = RMSNorm::from_weight_with_offset(
                get_tensor(&format!("layers.{}.ffn_norm.weight", i))?, eps, offset,
            );

            layers.push(TransformerBlock::from_weights_rms(attention, feed_forward, attention_norm, ffn_norm));
        }

        let norm = NormLayer::RMSNorm(RMSNorm::from_weight(get_tensor("norm.weight")?, eps));
        let output = Linear::from_weights(get_weight("output.weight")?, None)?;

        Ok(LlmModel {
            token_embedding,
            pos_embedding: None,
            embd_norm: None,
            ple: None,
            layers,
            norm,
            output,
            config: config.clone(),
        })
    }
}
