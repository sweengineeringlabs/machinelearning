//! gRPC server orchestration — wires the upstream
//! [`HandlerRegistryDispatcher`] and [`HealthService`] to the edge gRPC
//! ingress and runs the tonic listener.
//!
//! Returns the bound address and a shutdown channel so callers (the
//! `embed` binary, integration tests) can manage lifecycle.
//!
//! ## Wiring overview
//!
//! ```text
//!   TonicGrpcServer
//!         │
//!         ▼
//!   MethodPathRouter
//!     ├── /justembed.EmbedService/*  ──► HandlerRegistryDispatcher
//!     │                                    └── GrpcHandlerAdapter ──► EmbedHandler
//!     └── /grpc.health.v1.Health/*   ──► HealthService
//! ```
//!
//! At startup, the embed service is registered as `SERVING` in the
//! shared `HealthService`.  The dispatcher's aggregate health is what
//! we trust as the source of truth for the overall service status —
//! when `EmbedHandler::health_check` flips to `false` the next refresh
//! tick will mark the overall service `NOT_SERVING` and the named
//! `justembed.EmbedService` slot as well.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use edge_domain::HandlerRegistry;
use prost::Message;
use swe_edge_ingress_grpc::{
    GrpcHandlerAdapter, GrpcInboundError, HandlerRegistryDispatcher, HealthService,
    IngressTlsConfig, ServingStatus, TonicGrpcServer,
};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::api::config::EmbeddingGrpcConfig;
use crate::api::proto::{EmbedRequest, EmbedResponse};
use crate::core::grpc_dispatch::MethodPathRouter;
use crate::core::grpc_handler::EmbedHandler;
use crate::core::state::EmbeddingState;

/// Cadence at which the health refresh task polls the dispatcher and
/// pushes the aggregated status into the [`HealthService`].
///
/// Two seconds is short enough that a stuck embed handler is reflected
/// in `Health/Check` within one client retry budget, and long enough
/// that the model-load `health_check()` (which is currently O(1) but
/// could grow) does not dominate the runtime.
const HEALTH_REFRESH_INTERVAL: Duration = Duration::from_secs(2);

/// Service name reported by the standard `grpc.health.v1.Health.Check`
/// for the embed RPC service.  Convention is the fully-qualified proto
/// service name (no leading slash, no method).
pub const EMBED_HEALTH_SERVICE_NAME: &str = "justembed.EmbedService";

/// Running gRPC server: bound address + shutdown channel + serve task.
///
/// Drop the [`shutdown`](Self::shutdown) sender (or `send(())`) to stop
/// the server gracefully and await [`task`](Self::task) to observe the
/// result.  The dedicated health-refresh task is cancelled when the
/// shutdown trigger fires.
pub struct EmbedGrpcServer {
    /// Address actually bound (matters when `port = 0` was used).
    pub addr:     SocketAddr,
    /// Closes the listener when triggered or dropped.
    pub shutdown: oneshot::Sender<()>,
    /// Serving task — `await` to observe a clean exit or surface a
    /// bind/TLS error.
    pub task:     JoinHandle<Result<()>>,
}

/// Decode an [`EmbedRequest`] from raw protobuf bytes.
///
/// Surfaces decode failures as [`GrpcInboundError::InvalidArgument`] so
/// the wire shows `tonic::Code::InvalidArgument` to the caller.
fn decode_embed_request(bytes: &[u8]) -> Result<EmbedRequest, GrpcInboundError> {
    EmbedRequest::decode(bytes).map_err(|e| {
        GrpcInboundError::InvalidArgument(format!("decode EmbedRequest: {e}"))
    })
}

/// Encode an [`EmbedResponse`] to raw protobuf bytes.
///
/// `prost::Message::encode` to a `Vec<u8>` is infallible (the only
/// failure mode is a buffer too small, which `Vec<u8>` does not have),
/// so we use `expect` here — a panic in this code path implies a bug
/// in `prost`, not a runtime error.
fn encode_embed_response(resp: &EmbedResponse) -> Vec<u8> {
    let mut buf = Vec::with_capacity(resp.encoded_len());
    resp.encode(&mut buf).expect("EmbedResponse encode into Vec<u8> is infallible");
    buf
}

/// Spawn a gRPC server that exposes the registered embed handler and
/// the standard `grpc.health.v1.Health` service.
///
/// `cfg` controls bind, TLS, message-size cap, and the
/// `allow_unauthenticated` knob.  For now `allow_unauthenticated = true`
/// is honoured silently; flipping it to `false` while no auth interceptor
/// is wired up returns a startup error rather than appearing to enforce
/// auth that doesn't exist.
pub async fn start_grpc_server(
    state: Option<Arc<EmbeddingState>>,
    cfg:   &EmbeddingGrpcConfig,
) -> Result<EmbedGrpcServer> {
    if !cfg.allow_unauthenticated {
        anyhow::bail!(
            "[embedding.grpc].allow_unauthenticated = false but no auth interceptor \
             is currently registered.  Either set allow_unauthenticated = true or \
             extend the gRPC ingress with an auth handler before disabling it."
        );
    }

    // Build the byte-oriented registry and register the typed embed
    // handler through the upstream `GrpcHandlerAdapter`.  The adapter
    // forwards the inner handler's `id()` (the gRPC method path) to the
    // registry, so the dispatcher routes by URI path automatically.
    //
    // `state = None` runs the daemon in no-model mode: the service is
    // reachable, Embed calls return FailedPrecondition, health reports
    // not-serving. See `EmbedHandler` doc.
    let no_model = state.is_none();
    let registry: Arc<HandlerRegistry<Vec<u8>, Vec<u8>>> =
        Arc::new(HandlerRegistry::new());
    let embed_handler: Arc<dyn edge_domain::Handler<EmbedRequest, EmbedResponse>> =
        Arc::new(EmbedHandler::new(state));
    let adapter = GrpcHandlerAdapter::new(
        embed_handler,
        decode_embed_request,
        encode_embed_response,
    );
    registry.register(Arc::new(adapter));

    let dispatcher: Arc<HandlerRegistryDispatcher> =
        Arc::new(HandlerRegistryDispatcher::new(Arc::clone(&registry)));

    // Health service — overall + per-service slot.  Initial status
    // depends on whether a model is loaded:
    //   * loaded   → SERVING (model is fully loaded synchronously
    //                via `load_gguf` before we reach this code path)
    //   * no-model → NOT_SERVING (Embed RPCs will fail-fast with
    //                FailedPrecondition; operators see the degraded
    //                state through grpc.health.v1.Health/Check)
    // The refresh task below picks up handler-level health changes.
    let initial_status = if no_model {
        ServingStatus::NotServing
    } else {
        ServingStatus::Serving
    };
    let health = Arc::new(HealthService::new());
    health.set_overall_status(initial_status);
    health.set_status(EMBED_HEALTH_SERVICE_NAME, initial_status);

    // Wire the router that fans out by method-path prefix.
    let router = Arc::new(MethodPathRouter::new(
        dispatcher.clone() as Arc<dyn swe_edge_ingress_grpc::GrpcInbound>,
        health.clone()     as Arc<dyn swe_edge_ingress_grpc::GrpcInbound>,
    ));

    let bind = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&bind)
        .await
        .with_context(|| format!("gRPC bind {bind}"))?;
    let addr = listener.local_addr().context("gRPC local_addr")?;

    let mut server = TonicGrpcServer::new(bind.clone(), router)
        .with_max_message_size(cfg.max_message_bytes)
        .allow_unauthenticated(cfg.allow_unauthenticated);

    if let Some(tls) = build_tls_config(cfg)? {
        server = server.with_tls(tls);
    }

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    // Health refresh task — every `HEALTH_REFRESH_INTERVAL` we ask the
    // dispatcher for an aggregated health status and propagate it to
    // both the overall slot and the per-service slot.  Stops when the
    // shutdown signal fires.
    let refresh_health     = Arc::clone(&health);
    let refresh_dispatcher = Arc::clone(&dispatcher) as Arc<dyn swe_edge_ingress_grpc::GrpcInbound>;
    let (refresh_stop_tx, mut refresh_stop_rx) = oneshot::channel::<()>();
    let refresh_task = tokio::spawn(async move {
        let mut tick = tokio::time::interval(HEALTH_REFRESH_INTERVAL);
        // Skip the immediate first tick — startup status is already SERVING.
        tick.tick().await;
        loop {
            tokio::select! {
                _ = tick.tick() => {
                    let status = match refresh_dispatcher.health_check().await {
                        Ok(c) if c.healthy => ServingStatus::Serving,
                        _                  => ServingStatus::NotServing,
                    };
                    refresh_health.set_overall_status(status);
                    refresh_health.set_status(EMBED_HEALTH_SERVICE_NAME, status);
                }
                _ = &mut refresh_stop_rx => break,
            }
        }
    });

    let task = tokio::spawn(async move {
        let serve_result = server
            .serve_with_listener(listener, async move {
                let _ = shutdown_rx.await;
            })
            .await
            .map_err(|e| anyhow::anyhow!("gRPC server: {e}"));
        // Stop the refresh task before returning so its handle does
        // not leak past the server shutdown.
        let _ = refresh_stop_tx.send(());
        let _ = refresh_task.await;
        serve_result
    });

    log::info!(
        "embed: gRPC EmbedService on {}{} (health: grpc.health.v1.Health/Check)",
        addr,
        if cfg.tls.is_some() { " (TLS)" } else { "" }
    );

    Ok(EmbedGrpcServer { addr, shutdown: shutdown_tx, task })
}

/// Translate the daemon TLS config into the edge ingress shape.
fn build_tls_config(cfg: &EmbeddingGrpcConfig) -> Result<Option<IngressTlsConfig>> {
    let Some(tls) = cfg.tls.as_ref() else { return Ok(None); };
    if tls.cert_pem_path.is_empty() || tls.key_pem_path.is_empty() {
        anyhow::bail!(
            "[embedding.grpc.tls] requires both cert_pem_path and key_pem_path"
        );
    }
    let built = match tls.client_ca_pem_path.as_deref() {
        Some(ca) if !ca.is_empty() => {
            IngressTlsConfig::mtls(&tls.cert_pem_path, &tls.key_pem_path, ca)
        }
        _ => IngressTlsConfig::tls(&tls.cert_pem_path, &tls.key_pem_path),
    };
    Ok(Some(built))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: decode_embed_request — round-trips a valid request.
    #[test]
    fn test_decode_embed_request_round_trips_valid_proto_bytes() {
        let req = EmbedRequest {
            model: "m".to_string(),
            input: vec!["a".to_string(), "b".to_string()],
        };
        let mut buf = Vec::new();
        req.encode(&mut buf).unwrap();
        let decoded = decode_embed_request(&buf).expect("decode succeeds");
        assert_eq!(decoded.model, "m");
        assert_eq!(decoded.input, vec!["a".to_string(), "b".to_string()]);
    }

    /// @covers: decode_embed_request — malformed bytes surface as InvalidArgument.
    #[test]
    fn test_decode_embed_request_rejects_garbage_bytes_with_invalid_argument() {
        // 0xff prefix is not a valid varint+wire-type combination
        // for any tag in EmbedRequest, so prost rejects it.
        let bad = vec![0xff, 0xff, 0xff, 0xff];
        let err = decode_embed_request(&bad).expect_err("must fail");
        match err {
            GrpcInboundError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("decode EmbedRequest"),
                    "error must explain what failed to decode: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// @covers: encode_embed_response — round-trips with prost::Message::decode.
    #[test]
    fn test_encode_embed_response_round_trips_through_prost_decode() {
        let resp = EmbedResponse {
            model:      "m".to_string(),
            embeddings: vec![],
        };
        let bytes = encode_embed_response(&resp);
        let decoded = EmbedResponse::decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded.model, "m");
        assert_eq!(decoded.embeddings.len(), 0);
    }
}
