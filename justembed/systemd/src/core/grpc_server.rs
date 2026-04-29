//! gRPC server orchestration — wires the upstream
//! [`HandlerRegistryDispatcher`] to the edge gRPC ingress and runs the
//! tonic listener.
//!
//! Returns the bound address and a shutdown channel so callers (the
//! `embed` binary, integration tests) can manage lifecycle.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use edge_domain::HandlerRegistry;
use prost::Message;
use swe_edge_ingress_grpc::{
    GrpcHandlerAdapter, GrpcInboundError, HandlerRegistryDispatcher, IngressTlsConfig,
    TonicGrpcServer,
};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::api::config::EmbeddingGrpcConfig;
use crate::api::proto::{EmbedRequest, EmbedResponse};
use crate::core::grpc_handler::EmbedHandler;
use crate::core::state::EmbeddingState;

/// Running gRPC server: bound address + shutdown channel + serve task.
///
/// Drop the [`shutdown`](Self::shutdown) sender (or `send(())`) to stop the
/// server gracefully and await [`task`](Self::task) to observe the result.
pub struct EmbedGrpcServer {
    /// Address actually bound (matters when `port = 0` was used).
    pub addr:     SocketAddr,
    /// Closes the listener when triggered or dropped.
    pub shutdown: oneshot::Sender<()>,
    /// Serving task — `await` to observe a clean exit or surface a bind/TLS error.
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

/// Spawn a gRPC server that exposes the registered embed handler.
///
/// `cfg` controls bind, TLS, message-size cap, and the
/// `allow_unauthenticated` knob.  For now `allow_unauthenticated = true`
/// is honoured silently; flipping it to `false` while no auth interceptor
/// is wired up returns a startup error rather than appearing to enforce
/// auth that doesn't exist.
pub async fn start_grpc_server(
    state: Arc<EmbeddingState>,
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
    let registry: Arc<HandlerRegistry<Vec<u8>, Vec<u8>>> =
        Arc::new(HandlerRegistry::new());
    let embed_handler: Arc<dyn edge_domain::Handler<EmbedRequest, EmbedResponse>> =
        Arc::new(EmbedHandler::new(Arc::clone(&state)));
    let adapter = GrpcHandlerAdapter::new(
        embed_handler,
        decode_embed_request,
        encode_embed_response,
    );
    registry.register(Arc::new(adapter));

    let dispatcher: Arc<HandlerRegistryDispatcher> =
        Arc::new(HandlerRegistryDispatcher::new(registry));

    let bind = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&bind)
        .await
        .with_context(|| format!("gRPC bind {bind}"))?;
    let addr = listener.local_addr().context("gRPC local_addr")?;

    let mut server = TonicGrpcServer::new(bind.clone(), dispatcher)
        .with_max_message_size(cfg.max_message_bytes)
        .allow_unauthenticated(cfg.allow_unauthenticated);

    if let Some(tls) = build_tls_config(cfg)? {
        server = server.with_tls(tls);
    }

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        server
            .serve_with_listener(listener, async move {
                let _ = shutdown_rx.await;
            })
            .await
            .map_err(|e| anyhow::anyhow!("gRPC server: {e}"))
    });

    log::info!(
        "embed: gRPC EmbedService on {}{}",
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
