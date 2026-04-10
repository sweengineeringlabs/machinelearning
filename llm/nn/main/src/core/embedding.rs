//! Embedding layer implementation

use std::time::Instant;
use crate::api::error::NnResult;
use tensor_engine::Tensor;

/// Embedding layer that maps token indices to dense vectors
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Embedding weight matrix [num_embeddings, embedding_dim]
    pub weight: Tensor,
    /// Number of embeddings (vocabulary size)
    pub num_embeddings: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl Embedding {
    /// Create a new embedding layer with random initialization
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        // Standard normal initialization scaled by 0.02 (GPT-2 style)
        let weight = Tensor::randn(vec![num_embeddings, embedding_dim]).mul_scalar(0.02);

        Self {
            weight,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Create an embedding layer from existing weights
    pub fn from_weights(weight: Tensor) -> NnResult<Self> {
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(crate::api::error::NnError::InvalidConfig(
                "Embedding weight must be 2D".into(),
            ));
        }

        Ok(Self {
            num_embeddings: shape[0],
            embedding_dim: shape[1],
            weight,
        })
    }

    /// Forward pass: lookup embeddings for input indices
    ///
    /// Input shape: [...] (tensor of integer indices)
    /// Output shape: [..., embedding_dim]
    pub fn forward(&self, indices: &Tensor) -> NnResult<Tensor> {
        let _t = if log::log_enabled!(log::Level::Trace) { Some(Instant::now()) } else { None };

        let input_shape = indices.shape();
        let numel = indices.numel();

        // Gather embeddings
        let mut output_data = Vec::with_capacity(numel * self.embedding_dim);

        for idx_f32 in indices.iter() {
            let idx = idx_f32 as usize;
            if idx >= self.num_embeddings {
                return Err(crate::api::error::NnError::InvalidConfig(format!(
                    "Index {} out of bounds for embedding with {} entries",
                    idx, self.num_embeddings
                )));
            }

            // Get embedding vector for this index
            let embedding = self.weight.select(0, idx)?;
            output_data.extend(embedding.iter());
        }

        // Construct output shape: input_shape + [embedding_dim]
        let mut output_shape = input_shape.to_vec();
        output_shape.push(self.embedding_dim);

        let result = Tensor::from_vec(output_data, output_shape)?;
        if let Some(t) = _t {
            log::trace!("[perf] embedding::forward {:?}->{:?} {:.3}ms",
                input_shape, result.shape(), t.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(result)
    }
}

use crate::api::traits::PerLayerInput;
use crate::core::linear::Linear;
use crate::core::rms_norm::RMSNorm;

/// Per-Layer Embedding (PLE) for Gemma 4.
///
/// Pre-computes a combined per-layer input tensor from token identity and
/// context projection, then each decoder layer gates and projects its slice.
///
/// Pre-computation (once, before layers):
/// ```text
/// token_id   = embed_tokens_per_layer(ids) * sqrt(ple_dim)        → [B,S, L*D_ple]
///              reshape → [B, S, L, D_ple]
/// context    = model_projection(embeds) / sqrt(model_dim)          → [B,S, L*D_ple]
///              reshape → [B, S, L, D_ple]
///              projection_norm(context)
/// combined   = (context + token_id) / sqrt(2)                      → [B, S, L, D_ple]
/// ```
///
/// Per decoder layer i:
/// ```text
/// residual   = hidden
/// gate       = gelu(gate_proj[i](hidden))                          → [B,S, D_ple]
/// gated      = gate * combined[:,:,i,:]                            → [B,S, D_ple]
/// projected  = projection[i](gated)                                → [B,S, D_model]
/// normed     = post_norm[i](projected)
/// hidden     = residual + normed
/// ```
#[derive(Debug, Clone)]
pub struct PerLayerEmbedding {
    /// Shared embedding `[vocab, num_layers * ple_dim]`, scaled by sqrt(ple_dim)
    pub shared_embedding: Embedding,
    /// Per-layer dimension (slice width)
    pub ple_dim: usize,
    /// Context projection: model_dim → num_layers * ple_dim
    pub model_projection: Linear,
    /// Norm applied to context projection slices
    pub projection_norm: RMSNorm,
    /// Scale factor for model projection: 1/sqrt(model_dim)
    model_projection_scale: f32,
    /// Scale factor for embedding: sqrt(ple_dim)
    embedding_scale: f32,
    /// Scale factor for combination: 1/sqrt(2)
    input_scale: f32,
    /// Per-layer gate projections: model_dim → ple_dim
    pub gates: Vec<Linear>,
    /// Per-layer projections: ple_dim → model_dim
    pub projections: Vec<Linear>,
    /// Per-layer post-PLE norms on the projected output
    pub post_norms: Vec<RMSNorm>,
}

impl PerLayerEmbedding {
    /// Construct from pre-loaded weights.
    pub fn from_weights(
        shared_embedding: Embedding,
        ple_dim: usize,
        model_dim: usize,
        model_projection: Linear,
        projection_norm: RMSNorm,
        gates: Vec<Linear>,
        projections: Vec<Linear>,
        post_norms: Vec<RMSNorm>,
    ) -> NnResult<Self> {
        let n = gates.len();
        if projections.len() != n || post_norms.len() != n {
            return Err(crate::api::error::NnError::InvalidConfig(
                "PLE gates, projections, and post_norms must have the same length".into(),
            ));
        }
        let expected_dim = n * ple_dim;
        if shared_embedding.embedding_dim != expected_dim {
            return Err(crate::api::error::NnError::InvalidConfig(format!(
                "Shared PLE embedding dim {} != num_layers({}) * ple_dim({})",
                shared_embedding.embedding_dim, n, ple_dim,
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

    /// Pre-compute the combined per-layer input tensor.
    ///
    /// Call once before the layer loop, then pass slices to `inject()`.
    ///
    /// * `indices`      - token IDs `[B, S]`
    /// * `inputs_embeds` - token embeddings `[B, S, model_dim]` (after scaling)
    ///
    /// Returns `[B, S, num_layers, ple_dim]`.
    pub fn prepare(&self, indices: &Tensor, inputs_embeds: &Tensor) -> NnResult<Tensor> {
        let shape = indices.shape();
        let batch = shape[0];
        let seq = shape[1];
        let n_layers = self.gates.len();

        // Token identity: embed(ids) * sqrt(ple_dim) → [B, S, L*ple_dim]
        let token_id = self.shared_embedding.forward(indices)?
            .mul_scalar(self.embedding_scale);

        // Context projection: linear(embeds) / sqrt(model_dim) → [B, S, L*ple_dim]
        let context = self.model_projection.forward(inputs_embeds)?
            .mul_scalar(self.model_projection_scale);

        // Reshape both to [B, S, L, ple_dim] for per-layer norm
        let token_id = token_id.reshape(&[batch, seq, n_layers, self.ple_dim])?;
        let context = context.reshape(&[batch, seq, n_layers, self.ple_dim])?;

        // Apply RMSNorm on context (operates on last dim = ple_dim)
        let context = self.projection_norm.forward(&context)?;

        // Combine: (context + token_id) / sqrt(2)
        let combined = context.add(&token_id)?.mul_scalar(self.input_scale);

        Ok(combined)
    }

    /// Inject pre-computed per-layer input into hidden state for a given layer.
    ///
    /// * `per_layer_inputs` - pre-computed tensor from `prepare()`, `[B, S, L, ple_dim]`
    /// * `hidden`           - current hidden state `[B, S, model_dim]`
    /// * `layer`            - which layer index
    pub fn inject_prepared(&self, per_layer_inputs: &Tensor, hidden: &Tensor, layer: usize) -> NnResult<Tensor> {
        // Slice layer's input: [B, S, 1, ple_dim] → reshape → [B, S, ple_dim]
        let shape = per_layer_inputs.shape();
        let batch = shape[0];
        let seq = shape[1];
        let ple_input = per_layer_inputs.slice(2, layer, layer + 1)?
            .reshape(&[batch, seq, self.ple_dim])?;

        // Gate: gelu(gate_proj(hidden)) * ple_input
        let gate = self.gates[layer].forward(hidden)?.gelu();
        let gated = gate.mul(&ple_input)?;

        // Project back to model dim, norm, residual add
        let projected = self.projections[layer].forward(&gated)?;
        let normed = self.post_norms[layer].forward(&projected)?;
        Ok(hidden.add(&normed)?)
    }
}

impl PerLayerInput for PerLayerEmbedding {
    fn inject(&self, indices: &Tensor, hidden: &Tensor, layer: usize) -> NnResult<Tensor> {
        // Fallback: prepare + inject in one step (less efficient but correct)
        let dummy_embeds = hidden; // use hidden as proxy for inputs_embeds
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

        let shared = Embedding::new(vocab, num_layers * ple_dim);
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

        // Prepare once
        let prepared = ple.prepare(&indices, &hidden).unwrap();
        assert_eq!(prepared.shape(), &[1, 3, num_layers, ple_dim]);

        // Inject per layer
        let out0 = ple.inject_prepared(&prepared, &hidden, 0).unwrap();
        assert_eq!(out0.shape(), &[1, 3, model_dim]);

        let out1 = ple.inject_prepared(&prepared, &hidden, 1).unwrap();
        assert_eq!(out1.shape(), &[1, 3, model_dim]);
    }

    #[test]
    fn test_ple_dim_mismatch_rejected() {
        let shared = Embedding::new(100, 30); // 30 != 2 * 16
        let model_proj = Linear::new(64, 32);
        let proj_norm = RMSNorm::new(16, 1e-6);
        let gates = vec![Linear::new(64, 16), Linear::new(64, 16)];
        let projs = vec![Linear::new(16, 64), Linear::new(16, 64)];
        let norms = vec![RMSNorm::new(64, 1e-6), RMSNorm::new(64, 1e-6)];
        assert!(PerLayerEmbedding::from_weights(
            shared, 16, 64, model_proj, proj_norm, gates, projs, norms,
        ).is_err());
    }

    #[test]
    fn test_embedding_forward() {
        let embedding = Embedding::new(100, 32);
        let indices = Tensor::from_vec(vec![0.0, 5.0, 10.0, 50.0], vec![2, 2]).unwrap();
        let output = embedding.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[2, 2, 32]);
    }

    #[test]
    fn test_embedding_1d() {
        let embedding = Embedding::new(100, 64);
        let indices = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let output = embedding.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[3, 64]);
    }
}
