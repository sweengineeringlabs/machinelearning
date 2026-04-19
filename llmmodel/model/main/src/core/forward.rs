//! Transformer forward pass — prefill and autoregressive decoding.

use crate::api::error::{ModelError, ModelResult};
use crate::core::model::LlmModel;
use crate::core::per_layer_embedding::PerLayerInput;
use swe_llmmodel_layers::KVCache;
use swe_ml_tensor::{DType, Tensor, f32_vec_to_bytes};
use std::time::Instant;

impl LlmModel {
    /// Full-sequence forward pass (no KV cache). Used for prefill or scoring.
    pub fn forward_pass(&self, input_ids: &Tensor) -> ModelResult<Tensor> {
        let dump_inter = std::env::var("LLMSERV_DUMP_INTERMEDIATES").is_ok();
        let dump_stats = |label: &str, t: &Tensor| {
            if !dump_inter {
                return;
            }
            let v: Vec<f32> = t.iter().collect();
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut sumsq = 0.0_f32;
            let mut nan = 0usize;
            for x in &v {
                if x.is_nan() {
                    nan += 1;
                    continue;
                }
                if *x < min { min = *x; }
                if *x > max { max = *x; }
                sumsq += x * x;
            }
            let rms = (sumsq / v.len() as f32).sqrt();
            eprintln!(
                "[forward intermediate] {:32}  shape={:?}  rms={:.4}  min={:.4}  max={:.4}  range={:.4}  nan={}",
                label, t.shape(), rms, min, max, max - min, nan
            );
        };

        let x_emb = self.token_embedding.forward(input_ids)?;
        dump_stats("post_embed_lookup (raw)", &x_emb);
        let x_emb = if let Some(scale) = self.config.embedding_scale {
            x_emb.mul_scalar(scale)
        } else {
            x_emb
        };
        dump_stats("post_embed_scale", &x_emb);

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
            dump_stats("post_embd_norm", &x);
        }

        let ple_inputs = if let Some(ref ple) = self.ple {
            Some(ple.prepare(input_ids, &x)?)
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            if let (Some(ple), Some(pli)) = (&self.ple, &ple_inputs) {
                x = ple.inject_prepared(pli, &x, i)?;
            }
            x = layer.forward(&x)?;
            if dump_inter && (i < 3 || i == 12 || i == self.layers.len() - 1) {
                dump_stats(&format!("post_layer_{:02}", i), &x);
            }
        }

        let x = self.norm.forward(&x)?;
        dump_stats("post_final_norm", &x);
        let out = self.output.forward(&x)?;
        dump_stats("post_lm_head (logits)", &out);
        let out = if let Some(cap) = self.config.final_logit_softcapping {
            out.mul_scalar(1.0 / cap).tanh().mul_scalar(cap)
        } else {
            out
        };
        Ok(out)
    }

    /// Forward pass with KV cache for autoregressive decoding.
    pub fn forward_with_cache_pass(
        &self,
        input_ids: &Tensor,
        cache: &mut KVCache,
    ) -> ModelResult<Tensor> {
        let _t_total = if log::log_enabled!(log::Level::Debug) {
            Some(Instant::now())
        } else {
            None
        };

        let _t_emb = if log::log_enabled!(log::Level::Debug) {
            Some(Instant::now())
        } else {
            None
        };
        let x_emb = self.token_embedding.forward(input_ids)?;
        let x_emb = if let Some(scale) = self.config.embedding_scale {
            x_emb.mul_scalar(scale)
        } else {
            x_emb
        };
        let emb_ms = _t_emb
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[input_ids.ndim() - 1];
        let start_pos = cache.current_len;

        if start_pos + seq_len > self.config.max_seq_len {
            return Err(ModelError::Model(format!(
                "Sequence length {} exceeds maximum {} (start_pos={})",
                start_pos + seq_len,
                self.config.max_seq_len,
                start_pos
            )));
        }

        let mut x = if let Some(ref pos_emb) = self.pos_embedding {
            let mut pos_data = Vec::with_capacity(batch_size * seq_len);
            for _ in 0..batch_size {
                for i in 0..seq_len {
                    pos_data.push((start_pos + i) as f32);
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

        let ple_inputs = if let Some(ref ple) = self.ple {
            Some(ple.prepare(input_ids, &x)?)
        } else {
            None
        };

        let _t_layers = if log::log_enabled!(log::Level::Debug) {
            Some(Instant::now())
        } else {
            None
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let _t_layer = if log::log_enabled!(log::Level::Debug) {
                Some(Instant::now())
            } else {
                None
            };

            if let (Some(ple), Some(pli)) = (&self.ple, &ple_inputs) {
                x = ple.inject_prepared(pli, &x, i)?;
            }

            x = layer.forward_with_cache(&x, None, cache, i)?;
            if let Some(t) = _t_layer {
                log::debug!(
                    "[perf] model::forward layer={} total={:.3}ms",
                    i,
                    t.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        let layers_ms = _t_layers
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let _t_norm = if log::log_enabled!(log::Level::Debug) {
            Some(Instant::now())
        } else {
            None
        };
        let x = self.norm.forward(&x)?;
        let norm_ms = _t_norm
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        let _t_proj = if log::log_enabled!(log::Level::Debug) {
            Some(Instant::now())
        } else {
            None
        };
        let out = self.output.forward(&x)?;

        let out = if let Some(cap) = self.config.final_logit_softcapping {
            out.mul_scalar(1.0 / cap).tanh().mul_scalar(cap)
        } else {
            out
        };

        let proj_ms = _t_proj
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        if let Some(t) = _t_total {
            log::debug!(
                "[perf] model::forward embedding={:.3}ms layers={:.3}ms norm={:.3}ms projection={:.3}ms total={:.3}ms",
                emb_ms,
                layers_ms,
                norm_ms,
                proj_ms,
                t.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(out)
    }
}
