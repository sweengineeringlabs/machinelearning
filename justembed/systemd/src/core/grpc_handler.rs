//! `EmbedHandler` — domain handler that runs the shared embed loop and
//! returns proto-typed [`EmbedResponse`]s.
//!
//! Registered in [`edge_domain::HandlerRegistry`] under the fully-qualified
//! gRPC method path so the gRPC ingress can look it up by URI path.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use edge_domain::{Handler, HandlerError};

use crate::api::proto::{EmbedRequest, EmbedResponse, FloatVector, EMBED_METHOD_PATH};
use crate::core::embed::{EmbedError, embed_inputs};
use crate::core::state::EmbeddingState;

/// Domain handler bridging proto-typed [`EmbedRequest`] / [`EmbedResponse`]
/// to the shared embedding loop.
///
/// This is the gRPC analogue of the axum `embeddings` handler in
/// `core::router`.  Both call [`embed_inputs`] so the two transports are
/// guaranteed to produce the same vectors for the same inputs.
pub struct EmbedHandler {
    state: Arc<EmbeddingState>,
}

impl EmbedHandler {
    /// Construct an `EmbedHandler` over a shared model state.
    pub fn new(state: Arc<EmbeddingState>) -> Self {
        Self { state }
    }

    /// Stable id used both as the [`Handler::id`] and as the registry key.
    /// Kept as the gRPC method path so the ingress can dispatch by URI.
    pub const ID: &'static str = EMBED_METHOD_PATH;
}

#[async_trait]
impl Handler<EmbedRequest, EmbedResponse> for EmbedHandler {
    fn id(&self) -> &str { Self::ID }
    fn pattern(&self) -> &str { "Embed" }

    async fn execute(&self, req: EmbedRequest) -> Result<EmbedResponse, HandlerError> {
        let model_id = self.state.model_id.clone();
        let state    = Arc::clone(&self.state);
        let inputs   = req.input;

        let outcome = tokio::task::spawn_blocking(move || embed_inputs(state, inputs))
            .await
            .map_err(|e| HandlerError::ExecutionFailed(format!("task join: {e}")))?;

        let outcome = match outcome {
            Ok(o)                              => o,
            Err(EmbedError::EmptyInput)        => {
                return Err(HandlerError::InvalidRequest("input must not be empty".into()));
            }
            Err(EmbedError::Tokenization(msg)) => {
                return Err(HandlerError::InvalidRequest(format!("tokenization: {msg}")));
            }
            Err(other) => {
                return Err(HandlerError::ExecutionFailed(other.to_string()));
            }
        };

        Ok(EmbedResponse {
            model:      model_id,
            embeddings: outcome
                .vectors
                .into_iter()
                .map(|values| FloatVector { values })
                .collect(),
        })
    }

    async fn health_check(&self) -> bool { true }

    fn as_any(&self) -> &dyn Any { self }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: EmbedHandler::id — stable, equal to the proto method path.
    #[test]
    fn test_id_returns_grpc_method_path() {
        // Construction without a real model would require a fully-built
        // EmbeddingState; the id is defined as a const, so test the const
        // directly.  This guards against accidental rename drift.
        assert_eq!(EmbedHandler::ID, "/justembed.EmbedService/Embed");
    }
}
