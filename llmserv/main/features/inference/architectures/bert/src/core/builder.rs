use std::collections::HashMap;
use rustml_model::{
    LlmModel, ModelBuilder, ModelConfig, ModelError, ModelResult, TokenEmbedding,
};
use swe_llmmodel_layers::{
    Activation, FeedForward, LayerNorm, Linear, MultiHeadAttention, NormLayer, TransformerBlock,
};
use swe_ml_tensor::Tensor;

/// BERT architecture builder.
pub struct BertBuilder;

impl ModelBuilder for BertBuilder {
    fn remap_weights(&self, weights: HashMap<String, Tensor>, _config: &ModelConfig) -> HashMap<String, Tensor> {
        weights
    }

    fn build(&self, config: &ModelConfig, weights: HashMap<String, Tensor>) -> ModelResult<LlmModel> {
        let d_model = config.dim;
        let num_heads = config.n_heads;
        let num_layers = config.n_layers;
        let max_seq_len = config.max_seq_len;
        let eps = config.norm_eps;
        let position_encoding = config.position_encoding;
        let causal = config.causal;
        let rope_theta = config.rope_theta;

        let get_tensor = |key: &str| -> ModelResult<Tensor> {
            weights.get(key).ok_or_else(|| ModelError::Model(format!("Missing weight: {}", key))).and_then(|t| Ok(t.to_f32()?))
        };
        let get_tensor_opt = |key: &str| -> Option<Tensor> { weights.get(key).and_then(|t| t.to_f32().ok()) };

        let token_embedding = TokenEmbedding::from_weights(get_tensor("token_embedding.weight")?)?;
        let pos_embedding = if let Ok(w) = get_tensor("pos_embedding.weight") {
            Some(TokenEmbedding::from_weights(w)?)
        } else {
            Some(TokenEmbedding::new(max_seq_len, d_model))
        };

        let embd_norm = if let (Ok(w), Ok(b)) = (get_tensor("embd_norm.weight"), get_tensor("embd_norm.bias")) {
            Some(NormLayer::LayerNorm(LayerNorm::from_weights(w, b, eps)?))
        } else { None };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let q_proj = Linear::from_weights(get_tensor(&format!("layers.{i}.attention.q_proj.weight"))?, get_tensor_opt(&format!("layers.{i}.attention.q_proj.bias")))?;
            let k_proj = Linear::from_weights(get_tensor(&format!("layers.{i}.attention.k_proj.weight"))?, get_tensor_opt(&format!("layers.{i}.attention.k_proj.bias")))?;
            let v_proj = Linear::from_weights(get_tensor(&format!("layers.{i}.attention.v_proj.weight"))?, get_tensor_opt(&format!("layers.{i}.attention.v_proj.bias")))?;
            let out_proj = Linear::from_weights(get_tensor(&format!("layers.{i}.attention.out_proj.weight"))?, get_tensor_opt(&format!("layers.{i}.attention.out_proj.bias")))?;
            let attention = MultiHeadAttention::from_weights(d_model, num_heads, config.n_kv_heads, q_proj, k_proj, v_proj, out_proj, causal, position_encoding, max_seq_len, rope_theta)?;

            let up_proj = Linear::from_weights(get_tensor(&format!("layers.{i}.feed_forward.up_proj.weight"))?, get_tensor_opt(&format!("layers.{i}.feed_forward.up_proj.bias")))?;
            let down_proj = Linear::from_weights(get_tensor(&format!("layers.{i}.feed_forward.down_proj.weight"))?, get_tensor_opt(&format!("layers.{i}.feed_forward.down_proj.bias")))?;
            let feed_forward = FeedForward::from_weights_with_activation(up_proj, down_proj, Activation::Gelu);

            let attention_norm = LayerNorm::from_weights(get_tensor(&format!("layers.{i}.attention_norm.weight"))?, get_tensor(&format!("layers.{i}.attention_norm.bias"))?, eps)?;
            let ffn_norm = LayerNorm::from_weights(get_tensor(&format!("layers.{i}.ffn_norm.weight"))?, get_tensor(&format!("layers.{i}.ffn_norm.bias"))?, eps)?;

            layers.push(TransformerBlock::from_weights(attention, feed_forward, attention_norm, ffn_norm));
        }

        let norm = NormLayer::LayerNorm(LayerNorm::with_eps(d_model, eps));
        let output = Linear::from_weights(token_embedding.weight().clone(), None)?;

        Ok(LlmModel { token_embedding, pos_embedding, embd_norm, ple: None, layers, norm, output, config: config.clone() })
    }
}
