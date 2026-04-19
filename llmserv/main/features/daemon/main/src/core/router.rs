use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use tokio_stream::StreamExt;

use swe_ml_embedding::{L2Normalize, Normalize};
use rustml_generation::CompletionParams;
use rustml_inference_layers::PoolingStrategy;

use crate::api::error::DaemonError;
use crate::api::types::*;
use crate::core::state::AppState;

/// Build the per-request deadline from the daemon's configured timeout.
/// `None` preserves prior unbounded behavior; `Some(t)` ensures a stuck
/// forward pass cannot hold its throttle permit past `t`.
fn build_deadline(timeout: Option<Duration>) -> Option<Instant> {
    timeout.map(|d| Instant::now() + d)
}

/// Real tokenizer-backed length count. Replaces the prior
/// `split_whitespace().count()` estimate that systematically lied to
/// clients about token usage. Falls back to `0` on tokenizer error
/// rather than failing the request — the response itself is valid.
fn count_tokens(state: &AppState, text: &str) -> usize {
    state
        .model
        .tokenizer()
        .encode(text)
        .map(|ids| ids.len())
        .unwrap_or(0)
}

/// One item flowing from the blocking generator thread to the SSE
/// stream. `Done` carries the final usage so the SSE layer can emit a
/// terminal chunk before `[DONE]`, instead of the prior behavior of
/// silently omitting usage on streaming responses.
enum StreamItem {
    Token(String),
    Done(Usage),
}

/// Build the axum router with all endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(state)
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        model: Some(state.model.model_id().to_string()),
    })
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelInfo {
            id: state.model.model_id().to_string(),
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
    let permit = state
        .throttle
        .try_acquire()
        .ok_or_else(|| DaemonError::AtCapacity(state.throttle.capacity()))?;

    let owned_messages = collect_messages(&req);
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_k = req.top_k;
    let top_p = req.top_p;
    let repetition_penalty = req.repetition_penalty;
    let model_id = state.model.model_id().to_string();
    let deadline = build_deadline(state.request_timeout);

    // Tokenize prompt content on the async thread (cheap, no model forward
    // pass involved) so the count comes from the same tokenizer the model
    // will see — not a whitespace estimate. Excludes chat-template special
    // tokens, matching the convention of OpenAI-compatible servers.
    let prompt_tokens: usize = owned_messages
        .iter()
        .map(|(_, content)| count_tokens(&state, content))
        .sum();

    // Run inference on a blocking thread to avoid starving the tokio runtime.
    // `permit` moves into the closure; dropped when inference completes.
    let (output, completion_tokens) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut completer = state.model.open_text_completer();
        let params = CompletionParams {
            temperature,
            max_tokens,
            top_k,
            top_p,
            repetition_penalty,
            deadline,
        };

        let messages: Vec<(&str, &str)> = owned_messages
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();

        let start = Instant::now();
        let output: String = completer
            .complete_turn_stream(&messages, &params, &mut |_| true)
            .map_err(|e| DaemonError::GenerationFailed(e.to_string()))?;
        let elapsed = start.elapsed();

        let completion_tokens = count_tokens(&state, &output);

        log::info!(
            "Generated {} tokens in {:.2}s ({:.1} tok/s)",
            completion_tokens,
            elapsed.as_secs_f64(),
            completion_tokens as f64 / elapsed.as_secs_f64().max(1e-9)
        );

        Ok::<_, DaemonError>((output, completion_tokens))
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
    let permit = state
        .throttle
        .try_acquire()
        .ok_or_else(|| DaemonError::AtCapacity(state.throttle.capacity()))?;

    let completion_id = build_completion_id();
    let created = chrono::Utc::now().timestamp();
    let model_id = state.model.model_id().to_string();
    let owned_messages = collect_messages(&req);
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_k = req.top_k;
    let top_p = req.top_p;
    let repetition_penalty = req.repetition_penalty;
    let deadline = build_deadline(state.request_timeout);

    // Compute prompt tokens up-front using the real tokenizer so the value
    // is available for the terminal `Done` chunk regardless of how the
    // generation loop exits (early stop, deadline, client disconnect).
    let prompt_tokens: usize = owned_messages
        .iter()
        .map(|(_, content)| count_tokens(&state, content))
        .sum();

    let (tx, rx) = tokio::sync::mpsc::channel::<StreamItem>(64);

    let stream_state = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut completer = stream_state.model.open_text_completer();
        let params = CompletionParams {
            temperature,
            max_tokens,
            top_k,
            top_p,
            repetition_penalty,
            deadline,
        };

        let messages: Vec<(&str, &str)> = owned_messages
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();

        // Counted in the callback so the value reflects what the model
        // actually emitted, including partial runs ended by deadline or
        // client disconnect.
        let mut completion_tokens: usize = 0;
        let tokenizer = stream_state.model.tokenizer();
        let _ = completer.complete_turn_stream(&messages, &params, &mut |token_id| {
            completion_tokens += 1;
            match tokenizer.decode(&[token_id]) {
                Ok(piece) => tx.blocking_send(StreamItem::Token(piece)).is_ok(),
                Err(_) => true,
            }
        });

        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        };
        // Best-effort: if the receiver is gone (client disconnect), this
        // returns Err and we drop on the floor — nothing to do.
        let _ = tx.blocking_send(StreamItem::Done(usage));
    });

    let stream_id = completion_id.clone();
    let stream_model = model_id.clone();
    let token_stream =
        tokio_stream::wrappers::ReceiverStream::new(rx).map(move |item| {
            let chunk = match item {
                StreamItem::Token(piece) => ChatCompletionChunk {
                    id: stream_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: stream_model.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: Some(piece),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                },
                StreamItem::Done(usage) => ChatCompletionChunk {
                    id: stream_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: stream_model.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("stop".into()),
                    }],
                    usage: Some(usage),
                },
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

/// OpenAI-legacy text-completions endpoint. Takes a raw `prompt` and
/// generates continuation text WITHOUT applying the chat template.
/// Semantically matches the `llmserv_complete` FFI function, so HTTP
/// vs FFI can be compared on identical workloads.
async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, DaemonError> {
    if req.prompt.is_empty() {
        return Err(DaemonError::InvalidRequest("prompt must not be empty".into()));
    }
    if req.temperature < 0.0 {
        return Err(DaemonError::InvalidRequest(format!(
            "temperature must be >= 0.0, got {}",
            req.temperature
        )));
    }

    let permit = state
        .throttle
        .try_acquire()
        .ok_or_else(|| DaemonError::AtCapacity(state.throttle.capacity()))?;

    let prompt = req.prompt;
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_k = req.top_k;
    let top_p = req.top_p;
    let repetition_penalty = req.repetition_penalty;
    let model_id = state.model.model_id().to_string();
    let deadline = build_deadline(state.request_timeout);

    // Legacy /v1/completions takes a raw prompt — no chat template is
    // applied, so this token count is exact (matches what the model sees).
    let prompt_tokens = count_tokens(&state, &prompt);

    let (text, completion_tokens) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let mut completer = state.model.open_text_completer();
        let params = CompletionParams {
            temperature,
            max_tokens,
            top_k,
            top_p,
            repetition_penalty,
            deadline,
        };

        let start = Instant::now();
        let text = completer
            .complete(&prompt, &params)
            .map_err(|e| DaemonError::GenerationFailed(e.to_string()))?;
        let elapsed = start.elapsed();

        let completion_tokens = count_tokens(&state, &text);

        log::info!(
            "Generated (raw) {} tokens in {:.2}s ({:.1} tok/s)",
            completion_tokens,
            elapsed.as_secs_f64(),
            completion_tokens as f64 / elapsed.as_secs_f64().max(1e-9)
        );

        Ok::<_, DaemonError>((text, completion_tokens))
    })
    .await
    .map_err(|e| DaemonError::Internal(format!("Task join error: {}", e)))??;

    let response = CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4().simple()),
        object: "text_completion",
        created: chrono::Utc::now().timestamp(),
        model: model_id,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(response))
}

async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, DaemonError> {
    let inputs = req.input.into_vec();
    if inputs.is_empty() {
        return Err(DaemonError::InvalidRequest("input must not be empty".into()));
    }

    let model_id = state.model.model_id().to_string();
    let tokenizer = state.model.tokenizer();

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

    let permit = state
        .throttle
        .try_acquire()
        .ok_or_else(|| DaemonError::AtCapacity(state.throttle.capacity()))?;

    let state = Arc::clone(&state);
    let data = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let start = Instant::now();
        let mut results: Vec<(usize, Vec<f32>)> = Vec::with_capacity(all_token_ids.len());

        for (i, ids) in all_token_ids.iter().enumerate() {
            let mut vec = state
                .model
                .embed(ids, PoolingStrategy::Mean)
                .map_err(|e| DaemonError::GenerationFailed(format!("Embedding failed: {}", e)))?;
            L2Normalize
                .normalize(&mut vec)
                .map_err(|e| DaemonError::GenerationFailed(format!("Normalization failed: {}", e)))?;
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

