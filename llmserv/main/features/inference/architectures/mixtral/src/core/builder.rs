use std::collections::HashMap;
use rustml_model::{
    LlmModel, ModelBuilder, ModelConfig, ModelError, ModelResult, TokenEmbedding, WeightMap,
};
use rustml_inference_layers::{
    FeedForward, Linear, MoeLayer, MultiHeadAttention, NormLayer, RMSNorm, TransformerBlock,
};
use swe_ml_tensor::{DType, Tensor};

pub struct MixtralBuilder;

impl ModelBuilder for MixtralBuilder {
    fn remap_weights(&self, weights: HashMap<String, Tensor>, config: &ModelConfig) -> HashMap<String, Tensor> {
        let n_experts = config.num_local_experts.unwrap_or(8);
        WeightMap::mixtral(config.n_layers, n_experts).remap(weights)
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
        let num_experts = config.num_local_experts.unwrap_or(8);
        let num_experts_per_tok = config.num_experts_per_tok.unwrap_or(2);

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

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let q_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.q_proj.weight"))?, None)?;
            let k_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.k_proj.weight"))?, None)?;
            let v_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.v_proj.weight"))?, None)?;
            let out_proj = Linear::from_weights(get_weight(&format!("layers.{i}.attention.out_proj.weight"))?, None)?;
            let mut attention = MultiHeadAttention::from_weights(d_model, num_heads, config.n_kv_heads, q_proj, k_proj, v_proj, out_proj, causal, position_encoding, max_seq_len, rope_theta)?;
            if let Some(window) = config.sliding_window { attention.set_window_size(window); }

            let gate = Linear::from_weights(get_weight(&format!("layers.{i}.moe.gate.weight"))?, None)?;
            let mut experts = Vec::with_capacity(num_experts);
            for j in 0..num_experts {
                let gate_proj = Linear::from_weights(get_weight(&format!("layers.{i}.moe.experts.{j}.gate_proj.weight"))?, None)?;
                let up_proj = Linear::from_weights(get_weight(&format!("layers.{i}.moe.experts.{j}.up_proj.weight"))?, None)?;
                let down_proj = Linear::from_weights(get_weight(&format!("layers.{i}.moe.experts.{j}.down_proj.weight"))?, None)?;
                experts.push(FeedForward::from_weights_swiglu(up_proj, gate_proj, down_proj));
            }
            let moe_layer = MoeLayer::from_weights(gate, experts, num_experts_per_tok);

            let attention_norm = RMSNorm::from_weight(get_tensor(&format!("layers.{i}.attention_norm.weight"))?, eps);
            let ffn_norm = RMSNorm::from_weight(get_tensor(&format!("layers.{i}.ffn_norm.weight"))?, eps);
            let placeholder_ff = FeedForward::swiglu(d_model, config.hidden_dim, false);
            let mut block = TransformerBlock::from_weights_rms(attention, placeholder_ff, attention_norm, ffn_norm);
            block.moe = Some(moe_layer);
            layers.push(block);
        }

        let norm = NormLayer::RMSNorm(RMSNorm::from_weight(get_tensor("norm.weight")?, eps));
        let output = Linear::from_weights(get_weight("output.weight")?, None)?;

        Ok(LlmModel { token_embedding, pos_embedding: None, embd_norm: None, ple: None, layers, norm, output, config: config.clone() })
    }
}
