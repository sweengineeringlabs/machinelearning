use std::collections::HashMap;
use swe_ml_tensor::Tensor;

use crate::api::error::WeightResult;
use crate::api::traits::WeightMapper;

/// GPT-2 weight name mapper.
///
/// Maps HuggingFace GPT-2 weight names to architecture-internal names:
///
/// | HuggingFace                             | Internal                       |
/// |-----------------------------------------|--------------------------------|
/// | transformer.wte.weight                  | wte.weight                     |
/// | transformer.wpe.weight                  | wpe.weight                     |
/// | transformer.h.{i}.ln_1.weight           | blocks.{i}.ln_1.weight         |
/// | transformer.h.{i}.ln_1.bias             | blocks.{i}.ln_1.bias           |
/// | transformer.h.{i}.attn.c_attn.weight    | blocks.{i}.attn.c_attn.weight  |
/// | transformer.h.{i}.attn.c_attn.bias      | blocks.{i}.attn.c_attn.bias    |
/// | transformer.h.{i}.attn.c_proj.weight    | blocks.{i}.attn.c_proj.weight  |
/// | transformer.h.{i}.attn.c_proj.bias      | blocks.{i}.attn.c_proj.bias    |
/// | transformer.h.{i}.ln_2.weight           | blocks.{i}.ln_2.weight         |
/// | transformer.h.{i}.ln_2.bias             | blocks.{i}.ln_2.bias           |
/// | transformer.h.{i}.mlp.c_fc.weight       | blocks.{i}.mlp.c_fc.weight     |
/// | transformer.h.{i}.mlp.c_fc.bias         | blocks.{i}.mlp.c_fc.bias       |
/// | transformer.h.{i}.mlp.c_proj.weight     | blocks.{i}.mlp.c_proj.weight   |
/// | transformer.h.{i}.mlp.c_proj.bias       | blocks.{i}.mlp.c_proj.bias     |
/// | transformer.ln_f.weight                 | ln_f.weight                    |
/// | transformer.ln_f.bias                   | ln_f.bias                      |
#[derive(Debug, Default, Clone, Copy)]
pub struct Gpt2WeightMapper {
    pub n_layer: usize,
}

impl Gpt2WeightMapper {
    pub fn new(n_layer: usize) -> Self {
        Self { n_layer }
    }

    fn map_name(&self, hf_name: &str) -> Option<String> {
        let name = hf_name.strip_prefix("transformer.").unwrap_or(hf_name);

        if name.starts_with("h.") {
            let rest = name.strip_prefix("h.")?;
            Some(format!("blocks.{}", rest))
        } else {
            Some(name.to_string())
        }
    }

    /// Infer the number of transformer layers from HF weight names.
    pub fn detect_n_layer(weights: &HashMap<String, Tensor>) -> usize {
        weights
            .keys()
            .filter_map(|k| {
                let parts: Vec<&str> = k.split('.').collect();
                for (i, part) in parts.iter().enumerate() {
                    if *part == "h" && i + 1 < parts.len() {
                        return parts[i + 1].parse::<usize>().ok();
                    }
                }
                None
            })
            .max()
            .map(|n| n + 1)
            .unwrap_or(12)
    }
}

impl WeightMapper for Gpt2WeightMapper {
    fn map_weights(
        &self,
        weights: HashMap<String, Tensor>,
    ) -> WeightResult<HashMap<String, Tensor>> {
        let mut mapped = HashMap::new();

        for (name, tensor) in weights {
            if let Some(new_name) = self.map_name(&name) {
                // HuggingFace GPT-2 stores 2D weights as [in, out];
                // architecture expects [out, in].
                let tensor = if new_name.contains("c_attn.weight")
                    || new_name.contains("c_proj.weight")
                    || new_name.contains("c_fc.weight")
                {
                    if tensor.ndim() == 2 {
                        tensor.t().unwrap_or(tensor)
                    } else {
                        tensor
                    }
                } else {
                    tensor
                };

                mapped.insert(new_name, tensor);
            } else {
                mapped.insert(name, tensor);
            }
        }

        Ok(mapped)
    }

    fn expected_weights(&self) -> Vec<&'static str> {
        vec!["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_name_strips_transformer_prefix() {
        let mapper = Gpt2WeightMapper::new(12);
        assert_eq!(
            mapper.map_name("transformer.wte.weight"),
            Some("wte.weight".to_string())
        );
    }

    #[test]
    fn test_map_name_remaps_block_prefix() {
        let mapper = Gpt2WeightMapper::new(12);
        assert_eq!(
            mapper.map_name("transformer.h.0.ln_1.weight"),
            Some("blocks.0.ln_1.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("transformer.h.11.attn.c_attn.weight"),
            Some("blocks.11.attn.c_attn.weight".to_string())
        );
    }

    #[test]
    fn test_map_name_preserves_ln_f() {
        let mapper = Gpt2WeightMapper::new(12);
        assert_eq!(
            mapper.map_name("transformer.ln_f.weight"),
            Some("ln_f.weight".to_string())
        );
    }

    #[test]
    fn test_detect_n_layer_from_highest_block() {
        let mut weights = HashMap::new();
        weights.insert(
            "transformer.h.11.ln_1.weight".to_string(),
            Tensor::zeros(vec![768]),
        );
        weights.insert(
            "transformer.h.5.ln_1.weight".to_string(),
            Tensor::zeros(vec![768]),
        );

        assert_eq!(Gpt2WeightMapper::detect_n_layer(&weights), 12);
    }
}
