/// Map a HuggingFace tensor name to GGUF (llama.cpp) tensor name.
///
/// Supports Gemma and Llama-family naming conventions.
pub(crate) fn hf_to_gguf_name(hf_name: &str, arch: &str) -> String {
    // Strip multimodal prefix (Gemma 4 nests text model under model.language_model.)
    let name = hf_name
        .strip_prefix("model.language_model.")
        .map(|rest| format!("model.{rest}"))
        .unwrap_or_else(|| hf_name.to_string());
    let hf_name: &str = &name;

    // Extract layer index if present
    if let Some(rest) = hf_name.strip_prefix("model.layers.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_idx = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];
            let gguf_suffix = map_layer_suffix(suffix, arch);
            return format!("blk.{layer_idx}.{gguf_suffix}");
        }
    }

    // Global tensors
    match hf_name {
        "model.embed_tokens.weight" => "token_embd.weight".to_string(),
        "model.norm.weight" => "output_norm.weight".to_string(),
        "lm_head.weight" => "output.weight".to_string(),
        // Global PLE tensors (Gemma 4)
        "model.embed_tokens_per_layer.weight" => "ple_shared_embd.weight".to_string(),
        "model.per_layer_model_projection.weight" => "ple_model_proj.weight".to_string(),
        "model.per_layer_projection_norm.weight" => "ple_proj_norm.weight".to_string(),
        _ => {
            // Fallback: use the name as-is
            hf_name.to_string()
        }
    }
}

fn map_layer_suffix(suffix: &str, _arch: &str) -> String {
    match suffix {
        // Attention
        "self_attn.q_proj.weight" => "attn_q.weight".to_string(),
        "self_attn.k_proj.weight" => "attn_k.weight".to_string(),
        "self_attn.v_proj.weight" => "attn_v.weight".to_string(),
        "self_attn.o_proj.weight" => "attn_output.weight".to_string(),
        // QK norms (Gemma)
        "self_attn.q_norm.weight" => "attn_q_norm.weight".to_string(),
        "self_attn.k_norm.weight" => "attn_k_norm.weight".to_string(),
        // FFN / MLP
        "mlp.up_proj.weight" => "ffn_up.weight".to_string(),
        "mlp.down_proj.weight" => "ffn_down.weight".to_string(),
        "mlp.gate_proj.weight" => "ffn_gate.weight".to_string(),
        // Norms
        "input_layernorm.weight" => "attn_norm.weight".to_string(),
        "post_attention_layernorm.weight" => "post_attention_norm.weight".to_string(),
        "pre_feedforward_layernorm.weight" => "ffn_norm.weight".to_string(),
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight".to_string(),
        // Per-layer PLE (Gemma 4)
        "per_layer_input_gate.weight" => "ple_gate.weight".to_string(),
        "per_layer_projection.weight" => "ple_proj.weight".to_string(),
        "post_per_layer_input_norm.weight" => "post_ple_norm.weight".to_string(),
        // Layer scalar (Gemma 4)
        "layer_scalar" => "layer_scalar".to_string(),
        // Fallback
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_embedding() {
        assert_eq!(hf_to_gguf_name("model.embed_tokens.weight", "gemma4"), "token_embd.weight");
    }

    #[test]
    fn test_map_attention() {
        assert_eq!(
            hf_to_gguf_name("model.layers.0.self_attn.q_proj.weight", "gemma4"),
            "blk.0.attn_q.weight"
        );
    }

    #[test]
    fn test_map_ffn() {
        assert_eq!(
            hf_to_gguf_name("model.layers.5.mlp.up_proj.weight", "gemma4"),
            "blk.5.ffn_up.weight"
        );
    }

    #[test]
    fn test_map_output() {
        assert_eq!(hf_to_gguf_name("lm_head.weight", "gemma4"), "output.weight");
    }

    #[test]
    fn test_map_norms() {
        assert_eq!(
            hf_to_gguf_name("model.layers.0.input_layernorm.weight", "gemma4"),
            "blk.0.attn_norm.weight"
        );
        assert_eq!(hf_to_gguf_name("model.norm.weight", "gemma4"), "output_norm.weight");
    }

    #[test]
    fn test_map_ple_global() {
        assert_eq!(
            hf_to_gguf_name("model.language_model.embed_tokens_per_layer.weight", "gemma4"),
            "ple_shared_embd.weight"
        );
        assert_eq!(
            hf_to_gguf_name("model.language_model.per_layer_model_projection.weight", "gemma4"),
            "ple_model_proj.weight"
        );
    }

    #[test]
    fn test_map_ple_per_layer() {
        assert_eq!(
            hf_to_gguf_name("model.language_model.layers.0.per_layer_input_gate.weight", "gemma4"),
            "blk.0.ple_gate.weight"
        );
        assert_eq!(
            hf_to_gguf_name("model.language_model.layers.3.per_layer_projection.weight", "gemma4"),
            "blk.3.ple_proj.weight"
        );
    }
}
