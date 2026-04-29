//! gRPC server orchestration — wires the [`HandlerRegistry`] to the
//! edge gRPC ingress and runs the tonic listener.
//!
//! Returns the bound address and a shutdown channel so callers (the
//! `embed` binary, integration tests) can manage lifecycle.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use edge_domain::HandlerRegistry;
use swe_edge_ingress_grpc::{IngressTlsConfig, TonicGrpcServer};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::api::config::EmbeddingGrpcConfig;
use crate::api::proto::{EmbedRequest, EmbedResponse};
use crate::core::grpc_handler::EmbedHandler;
use crate::core::grpc_inbound::RegistryGrpcInbound;
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

    let registry: Arc<HandlerRegistry<EmbedRequest, EmbedResponse>> =
        Arc::new(HandlerRegistry::new());
    registry.register(Arc::new(EmbedHandler::new(Arc::clone(&state))));

    let inbound = Arc::new(RegistryGrpcInbound::new(Arc::clone(&registry)));

    let bind = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&bind)
        .await
        .with_context(|| format!("gRPC bind {bind}"))?;
    let addr = listener.local_addr().context("gRPC local_addr")?;

    let mut server = TonicGrpcServer::new(bind.clone(), inbound)
        .with_max_message_size(cfg.max_message_bytes);

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
