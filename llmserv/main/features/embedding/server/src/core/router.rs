use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};

use swe_ml_tensor::{DType, Tensor, f32_vec_to_bytes};
use rustml_inference_layers::PoolingStrategy;

use swe_ml_embedding::{
    EmbeddingData, EmbeddingUsage, EmbeddingsRequest, EmbeddingsResponse, L2Normalize, Normalize,
};
use crate::api::error::EmbeddingApiError;
use crate::core::state::EmbeddingState;

/// Build the embedding-only router.
pub fn build_embedding_router(state: Arc<EmbeddingState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/embeddings", post(embeddings))
        .with_state(state)
}

#[derive(serde::Serialize)]
struct HealthResponse {
    status: String,
    model: String,
}

async fn health(State(state): State<Arc<EmbeddingState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        model: state.model_id.clone(),
    })
}

async fn embeddings(
    State(state): State<Arc<EmbeddingState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, EmbeddingApiError> {
    let inputs = req.input.into_vec();
    if inputs.is_empty() {
        return Err(EmbeddingApiError("input must not be empty".into()));
    }

    let model_id = state.model_id.clone();
    let tokenizer = state.tokenizer.as_ref();

    let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(inputs.len());
    let mut total_tokens: usize = 0;
    for text in &inputs {
        let ids = tokenizer
            .encode(text)
            .map_err(|e| EmbeddingApiError(format!("Tokenization failed: {}", e)))?;
        total_tokens += ids.len();
        all_token_ids.push(ids);
    }

    let state = Arc::clone(&state);
    let data = tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let mut results: Vec<(usize, Vec<f32>)> = Vec::with_capacity(all_token_ids.len());

        for (i, ids) in all_token_ids.iter().enumerate() {
            let seq_len = ids.len();
            let input_data: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
            let input_tensor =
                Tensor::new(f32_vec_to_bytes(input_data), vec![1, seq_len], DType::F32);

            let embedding = state
                .model
                .embed(&input_tensor, PoolingStrategy::Mean)
                .map_err(|e| EmbeddingApiError(format!("Embedding failed: {}", e)))?;

            let mut vec: Vec<f32> = embedding.iter().collect();
            L2Normalize
                .normalize(&mut vec)
                .map_err(|e| EmbeddingApiError(format!("Normalization failed: {}", e)))?;
            results.push((i, vec));
        }

        log::info!(
            "Embedded {} input(s) ({} tokens) in {:.2}s",
            results.len(),
            all_token_ids.iter().map(|v| v.len()).sum::<usize>(),
            start.elapsed().as_secs_f64(),
        );

        Ok::<_, EmbeddingApiError>(results)
    })
    .await
    .map_err(|e| EmbeddingApiError(format!("Task join error: {}", e)))??;

    let response = EmbeddingsResponse {
        object: "list",
        data: data
            .into_iter()
            .map(|(index, embedding)| EmbeddingData {
                object: "embedding",
                index,
                embedding,
            })
            .collect(),
        model: model_id,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    Ok(Json(response))
}

