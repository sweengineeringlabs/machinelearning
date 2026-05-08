use std::collections::HashMap;
use swe_llmmodel_model::{
    LlmModel, ModelBuilder, ModelConfig, ModelError, ModelResult,
    TokenEmbedding, split_qkv, split_qkv_bias, map_gpt2_weights,
};
use swe_llmmodel_layers::{
    Activation, FeedForward, LayerNorm, Linear, MultiHeadAttention, NormLayer,
    PositionEncoding, TransformerBlock,
};
use swe_ml_tensor::Tensor;

/// GPT-2 architecture builder.
pub struct Gpt2Builder;

impl ModelBuilder for Gpt2Builder {
    fn remap_weights(
        &self,
        weights: HashMap<String, Tensor>,
        _config: &ModelConfig,
    ) -> HashMap<String, Tensor> {
        map_gpt2_weights(weights)
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
            weights.get(key)
                .ok_or_else(|| ModelError::Model(format!("Missing weight: {}", key)))
                .and_then(|t| Ok(t.to_f32()?))
        };
        let get_tensor_opt = |key: &str| -> Option<Tensor> {
            weights.get(key).and_then(|t| t.to_f32().ok())
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

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let c_attn_weight = get_tensor(&format!("layers.{}.attention.c_attn.weight", i))?
                .transpose(0, 1)?.contiguous()?;
            let c_attn_bias = get_tensor_opt(&format!("layers.{}.attention.c_attn.bias", i));

            let (q_w, k_w, v_w) = split_qkv(&c_attn_weight, d_model)?;
            let (q_b, k_b, v_b) = if let Some(bias) = c_attn_bias {
                let (qb, kb, vb) = split_qkv_bias(&bias, d_model)?;
                (Some(qb), Some(kb), Some(vb))
            } else {
                (None, None, None)
            };

            let q_proj = Linear::from_weights(q_w, q_b)?;
            let k_proj = Linear::from_weights(k_w, k_b)?;
            let v_proj = Linear::from_weights(v_w, v_b)?;

            let out_proj_weight = get_tensor(&format!("layers.{}.attention.out_proj.weight", i))?
                .transpose(0, 1)?.contiguous()?;
            let out_proj = Linear::from_weights(
                out_proj_weight,
                get_tensor_opt(&format!("layers.{}.attention.out_proj.bias", i)),
            )?;

            let attention = MultiHeadAttention::from_weights(
                d_model, num_heads, config.n_kv_heads,
                q_proj, k_proj, v_proj, out_proj,
                causal, position_encoding, max_seq_len, rope_theta,
            )?;

            let up_proj_weight = get_tensor(&format!("layers.{}.feed_forward.up_proj.weight", i))?
                .transpose(0, 1)?.contiguous()?;
            let up_proj = Linear::from_weights(
                up_proj_weight,
                get_tensor_opt(&format!("layers.{}.feed_forward.up_proj.bias", i)),
            )?;
            let down_proj_weight = get_tensor(&format!("layers.{}.feed_forward.down_proj.weight", i))?
                .transpose(0, 1)?.contiguous()?;
            let down_proj = Linear::from_weights(
                down_proj_weight,
                get_tensor_opt(&format!("layers.{}.feed_forward.down_proj.bias", i)),
            )?;
            let feed_forward = FeedForward::from_weights_with_activation(up_proj, down_proj, Activation::Gelu);

            let attention_norm = LayerNorm::from_weights(
                get_tensor(&format!("layers.{}.attention_norm.weight", i))?,
                get_tensor(&format!("layers.{}.attention_norm.bias", i))?,
                eps,
            )?;
            let ffn_norm = LayerNorm::from_weights(
                get_tensor(&format!("layers.{}.ffn_norm.weight", i))?,
                get_tensor(&format!("layers.{}.ffn_norm.bias", i))?,
                eps,
            )?;

            layers.push(TransformerBlock::from_weights(attention, feed_forward, attention_norm, ffn_norm));
        }

        let norm = NormLayer::LayerNorm(LayerNorm::from_weights(
            get_tensor("norm.weight")?,
            get_tensor("norm.bias")?,
            eps,
        )?);
        let output = Linear::from_weights(token_embedding.weight().clone(), None)?;

        Ok(LlmModel {
            token_embedding,
            pos_embedding,
            embd_norm: None,
            ple: None,
            layers,
            norm,
            output,
            config: config.clone(),
        })
    }
}
