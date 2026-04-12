use std::collections::HashMap;
use rustml_model::{
    LlmModel, ModelBuilder, ModelConfig, ModelError, ModelResult, TokenEmbedding, WeightMap,
};
use rustml_inference_layers::{
    Activation, FeedForward, LayerNorm, Linear, MultiHeadAttention, NormLayer, TransformerBlock,
};
use swe_ml_tensor::{DType, Tensor, f32_vec_to_bytes};

pub struct FalconBuilder;

impl ModelBuilder for FalconBuilder {
    fn remap_weights(&self, weights: HashMap<String, Tensor>, config: &ModelConfig) -> HashMap<String, Tensor> {
        WeightMap::falcon(config.n_layers).remap(weights)
    }

    fn build(&self, config: &ModelConfig, weights: HashMap<String, Tensor>) -> ModelResult<LlmModel> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_kv_heads = config.n_kv_heads.unwrap_or(num_heads);
        let num_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let position_encoding = config.position_encoding;
        let causal = config.causal;
        let rope_theta = config.rope_theta;
        let head_dim = d_model / num_heads;

        let get_tensor = |key: &str| -> ModelResult<Tensor> {
            weights.get(key).ok_or_else(|| ModelError::Model(format!("Missing weight: {}", key))).and_then(|t| Ok(t.to_f32()?))
        };
        let get_tensor_opt = |key: &str| -> Option<Tensor> { weights.get(key).and_then(|t| t.to_f32().ok()) };

        let token_embedding = TokenEmbedding::from_weights(get_tensor("token_embedding.weight")?)?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let qkv_weight = get_tensor(&format!("layers.{i}.attention.qkv.weight"))?;
            let qkv_bias = get_tensor_opt(&format!("layers.{i}.attention.qkv.bias"));
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let q_w = qkv_weight.slice(0, 0, q_dim)?;
            let k_w = qkv_weight.slice(0, q_dim, q_dim + kv_dim)?;
            let v_w = qkv_weight.slice(0, q_dim + kv_dim, q_dim + 2 * kv_dim)?;

            let (q_b, k_b, v_b) = if let Some(ref bias) = qkv_bias {
                let d: Vec<f32> = bias.iter().collect();
                (
                    Some(Tensor::new(f32_vec_to_bytes(d[..q_dim].to_vec()), vec![q_dim], DType::F32)),
                    Some(Tensor::new(f32_vec_to_bytes(d[q_dim..q_dim+kv_dim].to_vec()), vec![kv_dim], DType::F32)),
                    Some(Tensor::new(f32_vec_to_bytes(d[q_dim+kv_dim..].to_vec()), vec![kv_dim], DType::F32)),
                )
            } else { (None, None, None) };

            let q_proj = Linear::from_weights(q_w, q_b)?;
            let k_proj = Linear::from_weights(k_w, k_b)?;
            let v_proj = Linear::from_weights(v_w, v_b)?;
            let out_proj = Linear::from_weights(get_tensor(&format!("layers.{i}.attention.out_proj.weight"))?, get_tensor_opt(&format!("layers.{i}.attention.out_proj.bias")))?;
            let attention = MultiHeadAttention::from_weights(d_model, num_heads, config.n_kv_heads, q_proj, k_proj, v_proj, out_proj, causal, position_encoding, max_seq_len, rope_theta)?;

            let up_proj = Linear::from_weights(get_tensor(&format!("layers.{i}.feed_forward.up_proj.weight"))?, get_tensor_opt(&format!("layers.{i}.feed_forward.up_proj.bias")))?;
            let down_proj = Linear::from_weights(get_tensor(&format!("layers.{i}.feed_forward.down_proj.weight"))?, get_tensor_opt(&format!("layers.{i}.feed_forward.down_proj.bias")))?;
            let feed_forward = FeedForward::from_weights_with_activation(up_proj, down_proj, Activation::Gelu);

            let attention_norm = LayerNorm::from_weights(get_tensor(&format!("layers.{i}.attention_norm.weight"))?, get_tensor(&format!("layers.{i}.attention_norm.bias"))?, eps)?;
            let ffn_norm = LayerNorm::from_weights(get_tensor(&format!("layers.{i}.ffn_norm.weight"))?, get_tensor(&format!("layers.{i}.ffn_norm.bias"))?, eps)?;

            let mut block = TransformerBlock::from_weights(attention, feed_forward, attention_norm, ffn_norm);
            block.set_parallel_residual(config.parallel_residual.unwrap_or(true));
            layers.push(block);
        }

        let norm = NormLayer::LayerNorm(LayerNorm::from_weights(get_tensor("norm.weight")?, get_tensor("norm.bias")?, eps)?);
        let output = if let Ok(w) = get_tensor("output.weight") { Linear::from_weights(w, None)? } else { Linear::from_weights(token_embedding.weight().clone(), None)? };

        Ok(LlmModel { token_embedding, pos_embedding: None, embd_norm: None, ple: None, layers, norm, output, config: config.clone() })
    }
}
