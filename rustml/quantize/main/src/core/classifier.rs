use crate::api::types::TensorClass;

/// Classify a HuggingFace tensor name by its role in the model.
pub(crate) fn classify_tensor(name: &str) -> TensorClass {
    // Norms (always keep F32, 1D tensors)
    if name.contains("layernorm") || name.contains("layer_norm")
        || name.contains("norm") && name.ends_with(".weight")
        && !name.contains("proj")
    {
        return TensorClass::Norm;
    }

    // Embeddings
    if name.contains("embed_tokens") || name.contains("wte")
        || name.contains("wpe") || name.contains("token_embedding")
    {
        return TensorClass::Embedding;
    }

    // Output / LM head
    if name.contains("lm_head") || name == "output.weight" {
        return TensorClass::Output;
    }

    // Attention projections
    if name.contains("self_attn") || name.contains("attention")
        || name.contains("attn")
    {
        if name.contains("q_proj") || name.contains("k_proj")
            || name.contains("v_proj") || name.contains("o_proj")
            || name.contains("out_proj") || name.contains("qkv_proj")
        {
            return TensorClass::Attention;
        }
    }

    // Feed-forward / MLP projections
    if name.contains("mlp") || name.contains("feed_forward")
        || name.contains("ffn")
    {
        if name.contains("up_proj") || name.contains("down_proj")
            || name.contains("gate_proj") || name.contains("fc1")
            || name.contains("fc2") || name.contains("w1")
            || name.contains("w2") || name.contains("w3")
        {
            return TensorClass::FeedForward;
        }
    }

    // Gate (PLE, MoE)
    if name.contains("gate") && !name.contains("gate_proj") {
        return TensorClass::Gate;
    }

    // 2D weight tensors that we couldn't classify — treat as quantizable
    TensorClass::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_gemma4_attention() {
        assert_eq!(
            classify_tensor("model.layers.0.self_attn.q_proj.weight"),
            TensorClass::Attention
        );
        assert_eq!(
            classify_tensor("model.layers.5.self_attn.v_proj.weight"),
            TensorClass::Attention
        );
    }

    #[test]
    fn test_classify_gemma4_ffn() {
        assert_eq!(
            classify_tensor("model.layers.0.mlp.up_proj.weight"),
            TensorClass::FeedForward
        );
        assert_eq!(
            classify_tensor("model.layers.0.mlp.gate_proj.weight"),
            TensorClass::FeedForward
        );
    }

    #[test]
    fn test_classify_embedding() {
        assert_eq!(
            classify_tensor("model.embed_tokens.weight"),
            TensorClass::Embedding
        );
    }

    #[test]
    fn test_classify_norm() {
        assert_eq!(
            classify_tensor("model.layers.0.input_layernorm.weight"),
            TensorClass::Norm
        );
    }

    #[test]
    fn test_classify_output() {
        assert_eq!(
            classify_tensor("lm_head.weight"),
            TensorClass::Output
        );
    }
}
