use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};

use swe_ml_embedding::{
    EmbeddingData, EmbeddingUsage, EmbeddingsRequest, EmbeddingsResponse,
};
use crate::api::error::EmbeddingApiError;
use crate::core::embed::{EmbedError, embed_inputs};
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
    let inputs   = req.input.into_vec();
    let model_id = state.model_id.clone();

    let state_clone = Arc::clone(&state);
    let outcome = tokio::task::spawn_blocking(move || embed_inputs(state_clone, inputs))
        .await
        .map_err(|e| EmbeddingApiError(format!("Task join error: {}", e)))?
        .map_err(|e: EmbedError| EmbeddingApiError(e.to_string()))?;

    let response = EmbeddingsResponse {
        object: "list",
        data: outcome
            .vectors
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| EmbeddingData {
                object: "embedding",
                index,
                embedding,
            })
            .collect(),
        model: model_id,
        usage: EmbeddingUsage {
            prompt_tokens: outcome.total_tokens,
            total_tokens:  outcome.total_tokens,
        },
    };

    Ok(Json(response))
}
