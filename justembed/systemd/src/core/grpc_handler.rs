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
/// `state = None` is the **no-model mode** (issue #10): the daemon
/// boots without a GGUF, registers the gRPC service, but every Embed
/// call returns `HandlerError::FailedPrecondition` (gRPC code 9). The
/// health-check correspondingly reports unhealthy so the
/// `grpc.health.v1.Health/Check` aggregate reflects the degraded state.
/// Operators see "service up, model not loaded" rather than
/// "connection refused" — much clearer signal than a hard-exit at boot.
pub struct EmbedHandler {
    state: Option<Arc<EmbeddingState>>,
}

impl EmbedHandler {
    /// Construct an `EmbedHandler` over a shared model state.
    ///
    /// Pass `None` to run in no-model mode (Embed RPCs will return
    /// `FailedPrecondition`).
    pub fn new(state: Option<Arc<EmbeddingState>>) -> Self {
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
        // No-model mode: the daemon is reachable, the gRPC contract is
        // honoured, but no embedding can be served. Surface as a typed
        // FailedPrecondition (gRPC code 9 on the wire) with an
        // operator-actionable message.
        let state = match &self.state {
            Some(s) => s,
            None => {
                return Err(HandlerError::FailedPrecondition(
                    "no embedding model loaded; set [embedding.model].gguf_path \
                     in your XDG overlay (Linux: $XDG_CONFIG_HOME/llminference/, \
                     Windows: %APPDATA%\\llminference\\) and restart the daemon"
                        .to_string(),
                ));
            }
        };

        let model_id = state.model_id.clone();
        let state    = Arc::clone(state);
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

    /// Health: serving when a model is loaded, not-serving in no-model
    /// mode. The grpc_server's health-refresh task propagates this
    /// into both the overall slot and the per-service slot.
    async fn health_check(&self) -> bool { self.state.is_some() }

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

    /// @covers: EmbedHandler — no-model mode returns FailedPrecondition with
    /// an operator-actionable message that names the config knob to set.
    #[tokio::test]
    async fn test_execute_in_no_model_mode_returns_failed_precondition() {
        let handler = EmbedHandler::new(None);
        let req = EmbedRequest {
            model: "irrelevant".to_string(),
            input: vec!["hello".to_string()],
        };

        match handler.execute(req).await {
            Err(HandlerError::FailedPrecondition(msg)) => {
                assert!(
                    msg.contains("gguf_path"),
                    "error must name the config knob to set: {msg}"
                );
                assert!(
                    msg.contains("XDG"),
                    "error must point operators at the XDG overlay path: {msg}"
                );
            }
            other => panic!("expected FailedPrecondition in no-model mode, got {other:?}"),
        }
    }

    /// @covers: EmbedHandler::health_check — reflects whether a model is loaded.
    #[tokio::test]
    async fn test_health_check_reflects_no_model_mode() {
        let handler = EmbedHandler::new(None);
        assert!(
            !handler.health_check().await,
            "no-model mode must report unhealthy so the aggregate health surface \
             reflects the degraded state"
        );
    }
}
