/// Simple error type for the embedding API.
#[derive(Debug)]
pub struct EmbeddingApiError(pub String);

impl std::fmt::Display for EmbeddingApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl axum::response::IntoResponse for EmbeddingApiError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({ "error": { "message": self.0 } });
        (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(body),
        )
            .into_response()
    }
}
