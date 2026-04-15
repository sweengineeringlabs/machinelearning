use serde::{Deserialize, Serialize};

/// OpenAI-compatible chat completion request.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier (informational — the daemon serves one model at a time).
    #[serde(default)]
    pub model: String,

    /// Conversation messages.
    pub messages: Vec<ChatMessage>,

    /// Maximum tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-k sampling.
    pub top_k: Option<usize>,

    /// Nucleus (top-p) sampling.
    pub top_p: Option<f32>,

    /// Repetition penalty (1.0 = none).
    pub repetition_penalty: Option<f32>,

    /// Stream response tokens via SSE.
    #[serde(default)]
    pub stream: bool,
}

/// A single chat message.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// SSE streaming chunk (OpenAI-compatible delta format).
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    /// Populated only on the final chunk (matching OpenAI's
    /// `stream_options.include_usage` convention). Omitted on token
    /// chunks via `skip_serializing_if` so existing clients are unaffected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// OpenAI-legacy text completion request — `/v1/completions`.
/// Takes a raw prompt (no chat template applied), returns text in the
/// choices. Use this when you want apples-to-apples comparison against
/// `llmserv_complete` in the FFI library, or when the client is a
/// legacy SDK that uses the text-completions endpoint.
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: String,

    pub prompt: String,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    #[serde(default = "default_temperature")]
    pub temperature: f32,

    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

/// Response for GET /v1/models.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub owned_by: String,
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: Option<String>,
}

// Embedding API types — re-exported from swe-ml-embedding
pub use swe_ml_embedding::{
    EmbeddingData, EmbeddingInput, EmbeddingUsage, EmbeddingsRequest, EmbeddingsResponse,
};

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.8
}
