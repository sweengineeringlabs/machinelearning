use std::collections::HashMap;
use rustml_model::{
    LlmModel, ModelBuilder, ModelConfig, ModelError, ModelResult,
    TokenEmbedding, split_qkv,
};
use swe_llmmodel_layers::{
    FeedForward, LayerNorm, Linear, MultiHeadAttention, NormLayer, TransformerBlock,
};
use swe_ml_tensor::Tensor;

/// Nomic-BERT architecture builder.
///
/// Builds an LlmModel configured for Nomic-BERT: fused QKV split,
/// SwiGLU FFN, RoPE position encoding, bidirectional attention,
/// embedding LayerNorm.
pub struct NomicBertBuilder;

impl ModelBuilder for NomicBertBuilder {
    fn remap_weights(
        &self,
        weights: HashMap<String, Tensor>,
        _config: &ModelConfig,
    ) -> HashMap<String, Tensor> {
        // GGUF loader already remaps nomic-bert weights via load_and_remap_nomic_bert.
        // SafeTensors remapping would go here if needed.
        weights
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

        let token_embedding = TokenEmbedding::from_weights(get_tensor("token_embedding.weight")?)?;
        let pos_embedding: Option<TokenEmbedding> = None;

        let embd_norm = if let (Ok(w), Ok(b)) =
            (get_tensor("embd_norm.weight"), get_tensor("embd_norm.bias"))
        {
            Some(NormLayer::LayerNorm(LayerNorm::from_weights(w, b, eps)?))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let qkv = get_tensor(&format!("layers.{}.attention.qkv.weight", i))?;
            let (q_w, k_w, v_w) = split_qkv(&qkv, d_model)?;
            let q_proj = Linear::from_weights(q_w, None)?;
            let k_proj = Linear::from_weights(k_w, None)?;
            let v_proj = Linear::from_weights(v_w, None)?;
            let out_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.attention.out_proj.weight", i))?,
                None,
            )?;

            let attention = MultiHeadAttention::from_weights(
                d_model,
                num_heads,
                config.n_kv_heads,
                q_proj, k_proj, v_proj, out_proj,
                causal,
                position_encoding,
                max_seq_len,
                rope_theta,
            )?;

            let gate_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.feed_forward.gate_proj.weight", i))?,
                None,
            )?;
            let up_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.feed_forward.up_proj.weight", i))?,
                None,
            )?;
            let down_proj = Linear::from_weights(
                get_tensor(&format!("layers.{}.feed_forward.down_proj.weight", i))?,
                None,
            )?;
            let feed_forward = FeedForward::from_weights_swiglu(up_proj, gate_proj, down_proj);

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

            layers.push(TransformerBlock::from_weights(
                attention, feed_forward, attention_norm, ffn_norm,
            ));
        }

        let norm = NormLayer::LayerNorm(LayerNorm::with_eps(d_model, eps));
        let output = Linear::from_weights(token_embedding.weight().clone(), None)?;

        Ok(LlmModel {
            token_embedding,
            pos_embedding,
            embd_norm,
            ple: None,
            layers,
            norm,
            output,
            config: config.clone(),
        })
    }
}
