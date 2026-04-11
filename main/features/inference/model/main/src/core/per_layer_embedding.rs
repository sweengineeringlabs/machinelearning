//! PerLayerEmbedding — Gemma 4 per-layer input injection.

use crate::api::error::ModelResult;
use swe_ml_embedding::{DefaultEmbedding, Embed};
use swe_ml_tensor::Tensor;
use rustml_nn::{Linear, RMSNorm};

/// Per-layer input injection trait.
///
/// Given token indices and the current hidden state, produce an updated
/// hidden state that incorporates per-layer embedding information.
pub trait PerLayerInput {
    fn inject(&self, indices: &Tensor, hidden: &Tensor, layer: usize) -> ModelResult<Tensor>;
    fn num_layers(&self) -> usize;
}

/// Per-Layer Embedding (PLE) for Gemma 4.
///
/// Pre-computes a combined per-layer input tensor from token identity and
/// context projection, then each decoder layer gates and projects its slice.
#[derive(Debug, Clone)]
pub struct PerLayerEmbedding {
    pub shared_embedding: DefaultEmbedding,
    pub ple_dim: usize,
    pub model_projection: Linear,
    pub projection_norm: RMSNorm,
    model_projection_scale: f32,
    embedding_scale: f32,
    input_scale: f32,
    pub gates: Vec<Linear>,
    pub projections: Vec<Linear>,
    pub post_norms: Vec<RMSNorm>,
}

impl PerLayerEmbedding {
    pub fn from_weights(
        shared_embedding: DefaultEmbedding,
        ple_dim: usize,
        model_dim: usize,
        model_projection: Linear,
        projection_norm: RMSNorm,
        gates: Vec<Linear>,
        projections: Vec<Linear>,
        post_norms: Vec<RMSNorm>,
    ) -> ModelResult<Self> {
        let n = gates.len();
        if projections.len() != n || post_norms.len() != n {
            return Err(crate::api::error::ModelError::Model(
                "PLE gates, projections, and post_norms must have the same length".into(),
            ));
        }
        let expected_dim = n * ple_dim;
        if shared_embedding.embedding_dim() != expected_dim {
            return Err(crate::api::error::ModelError::Model(format!(
                "Shared PLE embedding dim {} != num_layers({}) * ple_dim({})",
                shared_embedding.embedding_dim(), n, ple_dim,
            )));
        }
        Ok(Self {
            shared_embedding,
            ple_dim,
            model_projection,
            projection_norm,
            model_projection_scale: (model_dim as f32).powf(-0.5),
            embedding_scale: (ple_dim as f32).sqrt(),
            input_scale: std::f32::consts::FRAC_1_SQRT_2,
            gates,
            projections,
            post_norms,
        })
    }

    pub fn prepare(&self, indices: &Tensor, inputs_embeds: &Tensor) -> ModelResult<Tensor> {
        let shape = indices.shape();
        let batch = shape[0];
        let seq = shape[1];
        let n_layers = self.gates.len();

        let token_id = self.shared_embedding.forward(indices)?
            .mul_scalar(self.embedding_scale);

        let context = self.model_projection.forward(inputs_embeds)?
            .mul_scalar(self.model_projection_scale);

        let token_id = token_id.reshape(&[batch, seq, n_layers, self.ple_dim])?;
        let context = context.reshape(&[batch, seq, n_layers, self.ple_dim])?;

        let context = self.projection_norm.forward(&context)?;

        let combined = context.add(&token_id)?.mul_scalar(self.input_scale);

        Ok(combined)
    }

    pub fn inject_prepared(&self, per_layer_inputs: &Tensor, hidden: &Tensor, layer: usize) -> ModelResult<Tensor> {
        let shape = per_layer_inputs.shape();
        let batch = shape[0];
        let seq = shape[1];
        let ple_input = per_layer_inputs.slice(2, layer, layer + 1)?
            .reshape(&[batch, seq, self.ple_dim])?;

        let gate = self.gates[layer].forward(hidden)?.gelu();
        let gated = gate.mul(&ple_input)?;

        let projected = self.projections[layer].forward(&gated)?;
        let normed = self.post_norms[layer].forward(&projected)?;
        Ok(hidden.add(&normed)?)
    }
}

impl PerLayerInput for PerLayerEmbedding {
    fn inject(&self, indices: &Tensor, hidden: &Tensor, layer: usize) -> ModelResult<Tensor> {
        let dummy_embeds = hidden;
        let per_layer_inputs = self.prepare(indices, dummy_embeds)?;
        self.inject_prepared(&per_layer_inputs, hidden, layer)
    }

    fn num_layers(&self) -> usize {
        self.gates.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ple_inject_shape() {
        let num_layers = 2;
        let ple_dim = 16;
        let model_dim = 64;
        let vocab = 100;

        let shared = DefaultEmbedding::new(vocab, num_layers * ple_dim);
        let model_proj = Linear::new(model_dim, num_layers * ple_dim);
        let proj_norm = RMSNorm::new(ple_dim, 1e-6);
        let gates = (0..num_layers).map(|_| Linear::new(model_dim, ple_dim)).collect();
        let projs = (0..num_layers).map(|_| Linear::new(ple_dim, model_dim)).collect();
        let norms = (0..num_layers).map(|_| RMSNorm::new(model_dim, 1e-6)).collect();

        let ple = PerLayerEmbedding::from_weights(
            shared, ple_dim, model_dim, model_proj, proj_norm, gates, projs, norms,
        ).unwrap();

        let indices = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![1, 3]).unwrap();
        let hidden = Tensor::randn(vec![1, 3, model_dim]);

        let prepared = ple.prepare(&indices, &hidden).unwrap();
        assert_eq!(prepared.shape(), &[1, 3, num_layers, ple_dim]);

        let out0 = ple.inject_prepared(&prepared, &hidden, 0).unwrap();
        assert_eq!(out0.shape(), &[1, 3, model_dim]);

        let out1 = ple.inject_prepared(&prepared, &hidden, 1).unwrap();
        assert_eq!(out1.shape(), &[1, 3, model_dim]);
    }

    #[test]
    fn test_ple_dim_mismatch_rejected() {
        let shared = DefaultEmbedding::new(100, 30);
        let model_proj = Linear::new(64, 32);
        let proj_norm = RMSNorm::new(16, 1e-6);
        let gates = vec![Linear::new(64, 16), Linear::new(64, 16)];
        let projs = vec![Linear::new(16, 64), Linear::new(16, 64)];
        let norms = vec![RMSNorm::new(64, 1e-6), RMSNorm::new(64, 1e-6)];
        assert!(PerLayerEmbedding::from_weights(
            shared, 16, 64, model_proj, proj_norm, gates, projs, norms,
        ).is_err());
    }
}
