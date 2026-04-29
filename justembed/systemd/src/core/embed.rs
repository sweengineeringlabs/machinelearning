//! Shared embedding loop — invoked by the gRPC `EmbedHandler`.
//!
//! Kept transport-agnostic so a future ingress (e.g. an in-process call
//! site, a benchmark harness) can call it directly without dragging the
//! tonic/proto layers along.
//!
//! The function is sync and CPU-bound — callers MUST wrap it in
//! [`tokio::task::spawn_blocking`] when invoking from an async context
//! to keep the runtime workers responsive.

use std::sync::Arc;
use std::time::Instant;

use swe_ml_embedding::{L2Normalize, Normalize};
use swe_ml_tensor::{DType, Tensor, f32_vec_to_bytes};
use swe_llmmodel_layers::PoolingStrategy;

use crate::core::state::EmbeddingState;

/// Failure modes surfaced by the embedding loop.
///
/// Mapped at the gRPC handler boundary:
///   * `EmptyInput` and `Tokenization` → `HandlerError::InvalidRequest`
///     (gRPC `InvalidArgument`).
///   * `Embed` and `Normalize`         → `HandlerError::ExecutionFailed`
///     (gRPC `Internal`, sanitised on the wire).
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// Caller sent an empty `input` field — refuse before any work runs.
    #[error("input must not be empty")]
    EmptyInput,
    /// Tokenizer rejected one of the inputs.
    #[error("tokenization failed: {0}")]
    Tokenization(String),
    /// Forward pass through the embedding model failed.
    #[error("embedding failed: {0}")]
    Embed(String),
    /// L2 normalisation failed (e.g. zero-norm vector).
    #[error("normalization failed: {0}")]
    Normalize(String),
}

/// Result of one embedding call: per-input vectors plus a token-count
/// summary that callers expose as part of their response shape.
#[derive(Debug)]
pub struct EmbedOutcome {
    /// One L2-normalised vector per input string, in input order.
    pub vectors:      Vec<Vec<f32>>,
    /// Total tokens across all inputs (sum of per-input lengths).
    pub total_tokens: usize,
}

/// Run the embedding loop end-to-end: tokenise → forward → L2-normalise.
///
/// Sync and CPU-bound on purpose — wrap in `spawn_blocking` from async
/// callers.  Tokenisation runs first (deterministic, cheap) so input
/// validation errors surface before the model is ever touched.
pub fn embed_inputs(
    state:  Arc<EmbeddingState>,
    inputs: Vec<String>,
) -> Result<EmbedOutcome, EmbedError> {
    if inputs.is_empty() {
        return Err(EmbedError::EmptyInput);
    }

    let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(inputs.len());
    let mut total_tokens: usize = 0;
    for text in &inputs {
        let ids = state
            .tokenizer
            .as_ref()
            .encode(text)
            .map_err(|e| EmbedError::Tokenization(e.to_string()))?;
        total_tokens += ids.len();
        all_token_ids.push(ids);
    }

    let start = Instant::now();
    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(all_token_ids.len());

    for ids in &all_token_ids {
        let seq_len    = ids.len();
        let input_data: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
        let input_tensor =
            Tensor::new(f32_vec_to_bytes(input_data), vec![1, seq_len], DType::F32);

        let embedding = state
            .model
            .embed(&input_tensor, PoolingStrategy::Mean)
            .map_err(|e| EmbedError::Embed(e.to_string()))?;

        let mut vec: Vec<f32> = embedding.iter().collect();
        L2Normalize
            .normalize(&mut vec)
            .map_err(|e| EmbedError::Normalize(e.to_string()))?;
        vectors.push(vec);
    }

    log::info!(
        "Embedded {} input(s) ({} tokens) in {:.2}s",
        vectors.len(),
        total_tokens,
        start.elapsed().as_secs_f64(),
    );

    Ok(EmbedOutcome { vectors, total_tokens })
}
