//! Embedding extraction — encode() and embed() methods.

use crate::api::error::{ModelError, ModelResult};
use crate::core::model::LlmModel;
use rustml_inference_layers::PoolingStrategy;
use swe_ml_tensor::{DType, Tensor, f32_vec_to_bytes};

impl LlmModel {
    /// Encode input tokens to hidden states: `[B, S]` → `[B, S, dim]`.
    ///
    /// Runs the full transformer stack without the output projection head.
    /// Used by `embed()` for dense vector extraction.
    pub fn encode(&self, input_ids: &Tensor) -> ModelResult<Tensor> {
        let x_emb = self.token_embedding.forward(input_ids)?;
        let x_emb = if let Some(scale) = self.config.embedding_scale {
            x_emb.mul_scalar(scale)
        } else {
            x_emb
        };

        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[input_ids.ndim() - 1];

        if seq_len > self.config.max_seq_len {
            return Err(ModelError::Model(format!(
                "Sequence length {} exceeds maximum {}",
                seq_len, self.config.max_seq_len
            )));
        }

        let mut x = if let Some(ref pos_emb) = self.pos_embedding {
            let mut pos_data = Vec::with_capacity(batch_size * seq_len);
            for _ in 0..batch_size {
                for i in 0..seq_len {
                    pos_data.push(i as f32);
                }
            }
            let pos_bytes = f32_vec_to_bytes(pos_data);
            let pos_ids = Tensor::new(pos_bytes, vec![batch_size, seq_len], DType::F32);
            let p_emb = pos_emb.forward(&pos_ids)?;
            x_emb.add(&p_emb)?
        } else {
            x_emb
        };

        if let Some(ref embd_norm) = self.embd_norm {
            x = embd_norm.forward(&x)?;
        }

        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        Ok(self.norm.forward(&x)?)
    }

    /// Embed input tokens to a single dense vector per batch item: `[B, S]` → `[B, dim]`.
    ///
    /// Runs `encode()` then pools across the sequence dimension.
    pub fn embed(&self, input_ids: &Tensor, strategy: PoolingStrategy) -> ModelResult<Tensor> {
        let hidden = self.encode(input_ids)?;
        pool_hidden_states(&hidden, strategy)
    }
}

/// Pool hidden states across the sequence dimension.
pub(crate) fn pool_hidden_states(hidden: &Tensor, strategy: PoolingStrategy) -> ModelResult<Tensor> {
    let shape = hidden.shape();
    if shape.len() != 3 {
        return Err(ModelError::Model(format!(
            "pool_hidden_states: expected [B, S, dim] tensor, got {:?}",
            shape
        )));
    }
    let batch_size = shape[0];
    let seq_len = shape[1];
    let dim = shape[2];

    let data: Vec<f32> = hidden.iter().collect();
    let mut output = vec![0.0f32; batch_size * dim];

    match strategy {
        PoolingStrategy::Cls => {
            for b in 0..batch_size {
                let src = b * seq_len * dim;
                let dst = b * dim;
                output[dst..dst + dim].copy_from_slice(&data[src..src + dim]);
            }
        }
        PoolingStrategy::Mean => {
            let scale = 1.0 / seq_len as f32;
            for b in 0..batch_size {
                let dst = b * dim;
                for s in 0..seq_len {
                    let src = (b * seq_len + s) * dim;
                    for d in 0..dim {
                        output[dst + d] += data[src + d];
                    }
                }
                for d in 0..dim {
                    output[dst + d] *= scale;
                }
            }
        }
        PoolingStrategy::Max => {
            for b in 0..batch_size {
                let dst = b * dim;
                let src0 = b * seq_len * dim;
                output[dst..dst + dim].copy_from_slice(&data[src0..src0 + dim]);
                for s in 1..seq_len {
                    let src = (b * seq_len + s) * dim;
                    for d in 0..dim {
                        if data[src + d] > output[dst + d] {
                            output[dst + d] = data[src + d];
                        }
                    }
                }
            }
        }
    }

    let bytes = f32_vec_to_bytes(output);
    Ok(Tensor::new(bytes, vec![batch_size, dim], DType::F32))
}
