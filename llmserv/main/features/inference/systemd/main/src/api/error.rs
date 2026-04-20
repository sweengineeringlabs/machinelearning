use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

pub type DaemonResult<T> = Result<T, DaemonError>;

#[derive(Error, Debug)]
pub enum DaemonError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Model loading failed: {0}")]
    LoadFailed(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Server at capacity (max concurrent={0})")]
    AtCapacity(usize),
}

impl IntoResponse for DaemonError {
    fn into_response(self) -> Response {
        let status = match &self {
            DaemonError::ModelNotLoaded(_) => StatusCode::SERVICE_UNAVAILABLE,
            DaemonError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            DaemonError::GenerationFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            DaemonError::LoadFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            DaemonError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            DaemonError::AtCapacity(_) => StatusCode::SERVICE_UNAVAILABLE,
        };

        let body = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": match &self {
                    DaemonError::ModelNotLoaded(_) => "model_not_loaded",
                    DaemonError::InvalidRequest(_) => "invalid_request_error",
                    DaemonError::GenerationFailed(_) => "generation_error",
                    DaemonError::LoadFailed(_) => "load_error",
                    DaemonError::Internal(_) => "internal_error",
                    DaemonError::AtCapacity(_) => "at_capacity",
                },
            }
        });

        (status, axum::Json(body)).into_response()
    }
}
