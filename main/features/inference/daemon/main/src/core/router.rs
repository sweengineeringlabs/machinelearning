use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use tokio_stream::StreamExt;

use swe_ml_tensor::{DType, Tensor, f32_vec_to_bytes};
use swe_ml_embedding::l2_normalize;
use rustml_inference_layers::PoolingStrategy;

use crate::api::error::DaemonError;
use crate::api::types::*;
use crate::core::state::AppState;

/// Build the axum router with all endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(state)
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        model: Some(state.bundle.model_id.clone()),
    })
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelInfo {
            id: state.bundle.model_id.clone(),
            object: "model",
            owned_by: "local".into(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, DaemonError> {
    validate_request(&req)?;

    if req.stream {
        return handle_streaming(state, req).await;
    }

    handle_blocking(state, req).await
}

fn validate_request(req: &ChatCompletionRequest) -> Result<(), DaemonError> {
    if req.messages.is_empty() {
        return Err(DaemonError::InvalidRequest(
            "messages array must not be empty".into(),
        ));
    }
    if req.temperature < 0.0 {
        return Err(DaemonError::InvalidRequest(format!(
            "temperature must be >= 0.0, got {}",
            req.temperature
        )));
    }
    if let Some(k) = req.top_k {
        if k == 0 {
            return Err(DaemonError::InvalidRequest("top_k must be > 0".into()));
        }
    }
    if let Some(p) = req.top_p {
        if !(0.0..=1.0).contains(&p) || p == 0.0 {
            return Err(DaemonError::InvalidRequest(format!(
                "top_p must be in (0.0, 1.0], got {}",
                p
            )));
        }
    }
    if let Some(rp) = req.repetition_penalty {
        if rp <= 0.0 {
            return Err(DaemonError::InvalidRequest(format!(
                "repetition_penalty must be > 0.0, got {}",
                rp
            )));
        }
    }
    Ok(())
}

/// Convert request messages into owned (role, content) pairs for the generator.
fn collect_messages(req: &ChatCompletionRequest) -> Vec<(String, String)> {
    req.messages
        .iter()
        .filter(|m| m.role == "user" || m.role == "assistant")
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect()
}

fn build_completion_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4().simple())
}

async fn handle_blocking(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<axum::response::Response, DaemonError> {
    let owned_messages = collect_messages(&req);
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_k = req.top_k;
    let top_p = req.top_p;
    let repetition_penalty = req.repetition_penalty;
    let model_id = state.bundle.model_id.clone();

    // Run inference on a blocking thread to avoid starving the tokio runtime.
    let (output, prompt_tokens, completion_tokens) = tokio::task::spawn_blocking(move || {
        let mut generator = state.bundle.build_generator(temperature);
        if let Some(k) = top_k {
            generator = generator.with_top_k(k);
        }
        if let Some(p) = top_p {
            generator = generator.with_top_p(p);
        }
        if let Some(rp) = repetition_penalty {
            generator = generator.with_repetition_penalty(rp);
        }

        let messages: Vec<(&str, &str)> = owned_messages
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();

        let start = Instant::now();
        let output: String = generator
            .generate_turn_stream(&messages, max_tokens, |_| true)
            .map_err(|e| DaemonError::GenerationFailed(e.to_string()))?;
        let elapsed = start.elapsed();

        let prompt_tokens = owned_messages
            .iter()
            .map(|(_, c)| c.split_whitespace().count())
            .sum::<usize>();
        let completion_tokens = output.split_whitespace().count();

        log::info!(
            "Generated {} tokens in {:.2}s ({:.1} tok/s)",
            completion_tokens,
            elapsed.as_secs_f64(),
            completion_tokens as f64 / elapsed.as_secs_f64().max(1e-9)
        );

        Ok::<_, DaemonError>((output, prompt_tokens, completion_tokens))
    })
    .await
    .map_err(|e| DaemonError::Internal(format!("Task join error: {}", e)))??;

    let response = ChatCompletionResponse {
        id: build_completion_id(),
        object: "chat.completion",
        created: chrono::Utc::now().timestamp(),
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: output,
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(response).into_response())
}

async fn handle_streaming(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<axum::response::Response, DaemonError> {
    let completion_id = build_completion_id();
    let created = chrono::Utc::now().timestamp();
    let model_id = state.bundle.model_id.clone();
    let owned_messages = collect_messages(&req);
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_k = req.top_k;
    let top_p = req.top_p;
    let repetition_penalty = req.repetition_penalty;

    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);

    let stream_state = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let mut generator = stream_state.bundle.build_generator(temperature);
        if let Some(k) = top_k {
            generator = generator.with_top_k(k);
        }
        if let Some(p) = top_p {
            generator = generator.with_top_p(p);
        }
        if let Some(rp) = repetition_penalty {
            generator = generator.with_repetition_penalty(rp);
        }

        let messages: Vec<(&str, &str)> = owned_messages
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();

        let tokenizer = stream_state.bundle.tokenizer.as_ref();
        let _ = generator.generate_turn_stream(&messages, max_tokens, |token_id| {
            match tokenizer.decode(&[token_id]) {
                Ok(piece) => tx.blocking_send(piece).is_ok(),
                Err(_) => true,
            }
        });
    });

    let token_stream =
        tokio_stream::wrappers::ReceiverStream::new(rx).map(move |piece| {
            let chunk = ChatCompletionChunk {
                id: completion_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_id.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: ChatDelta {
                        role: None,
                        content: Some(piece),
                    },
                    finish_reason: None,
                }],
            };
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            Ok::<_, Infallible>(Event::default().data(json))
        });

    let done = stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
    let full_stream = token_stream.chain(done);

    Ok(Sse::new(full_stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, DaemonError> {
    let inputs = req.input.into_vec();
    if inputs.is_empty() {
        return Err(DaemonError::InvalidRequest("input must not be empty".into()));
    }

    let model_id = state.bundle.model_id.clone();
    let tokenizer = state.bundle.tokenizer.as_ref();

    // Tokenize all inputs upfront (cheap, keep on async thread).
    let mut all_token_ids: Vec<Vec<u32>> = Vec::with_capacity(inputs.len());
    let mut total_tokens: usize = 0;
    for text in &inputs {
        let ids = tokenizer
            .encode(text)
            .map_err(|e| DaemonError::InvalidRequest(format!("Tokenization failed: {}", e)))?;
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
            let input_tensor = Tensor::new(f32_vec_to_bytes(input_data), vec![1, seq_len], DType::F32);

            let embedding = state
                .bundle
                .model
                .embed(&input_tensor, PoolingStrategy::Mean)
                .map_err(|e| DaemonError::GenerationFailed(format!("Embedding failed: {}", e)))?;

            let mut vec: Vec<f32> = embedding.iter().collect();
            l2_normalize(&mut vec);
            results.push((i, vec));
        }

        let elapsed = start.elapsed();
        log::info!(
            "Embedded {} input(s) ({} tokens) in {:.2}s",
            results.len(),
            all_token_ids.iter().map(|v| v.len()).sum::<usize>(),
            elapsed.as_secs_f64(),
        );

        Ok::<_, DaemonError>(results)
    })
    .await
    .map_err(|e| DaemonError::Internal(format!("Task join error: {}", e)))??;

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

