//! `MethodPathRouter` — routes inbound gRPC calls by method path prefix.
//!
//! The upstream `swe-edge-ingress-grpc` ships two `GrpcInbound` impls
//! that we want to use side-by-side:
//!
//! * [`HandlerRegistryDispatcher`] — dispatches user-defined RPCs by
//!   their fully-qualified method path (here: `/justembed.EmbedService/Embed`).
//! * [`HealthService`] — owns the standard `grpc.health.v1.Health` service
//!   (`Check` + `Watch`).
//!
//! [`TonicGrpcServer`] takes a single `Arc<dyn GrpcInbound>`, so we need
//! a tiny composite that fans out to the right one based on the request
//! method path.  Anything under `/grpc.health.v1.Health/...` goes to
//! `HealthService`; everything else goes to the user dispatcher.
//!
//! Why a router instead of registering a `GrpcHandlerAdapter` for Check
//! in the same registry as Embed:
//!
//! * `HealthService` parses its own proto wire format and owns
//!   per-service status state — it is *not* a `Handler<Req, Resp>`.
//! * `Watch` is a server-streaming RPC with non-trivial cancellation
//!   and broadcast semantics; the registry adapter only models unary
//!   request/response pairs.
//!
//! [`HandlerRegistryDispatcher`]: swe_edge_ingress_grpc::HandlerRegistryDispatcher
//! [`HealthService`]: swe_edge_ingress_grpc::HealthService
//! [`TonicGrpcServer`]: swe_edge_ingress_grpc::TonicGrpcServer

use std::sync::Arc;

use futures::future::BoxFuture;
use swe_edge_ingress_grpc::{
    GrpcHealthCheck, GrpcInbound, GrpcInboundResult, GrpcMessageStream, GrpcMetadata,
    GrpcRequest, GrpcResponse,
};

/// Method-path prefix that selects the standard health service.
///
/// Both `Check` and `Watch` live under this prefix:
///
/// * `/grpc.health.v1.Health/Check`
/// * `/grpc.health.v1.Health/Watch`
pub const HEALTH_SERVICE_PREFIX: &str = "/grpc.health.v1.Health/";

/// `MethodPathRouter` — fans incoming calls out to one of two backing
/// `GrpcInbound`s based on the request method path.
///
/// Calls whose method starts with [`HEALTH_SERVICE_PREFIX`] go to
/// `health`; every other call goes to `default`.  This is the minimum
/// glue needed to expose both the user RPC service and the standard
/// `grpc.health.v1.Health` service through a single
/// [`TonicGrpcServer`](swe_edge_ingress_grpc::TonicGrpcServer).
pub struct MethodPathRouter {
    default: Arc<dyn GrpcInbound>,
    health:  Arc<dyn GrpcInbound>,
}

impl MethodPathRouter {
    /// Build a router with `default` as the catch-all and `health` as
    /// the destination for any method under [`HEALTH_SERVICE_PREFIX`].
    pub fn new(default: Arc<dyn GrpcInbound>, health: Arc<dyn GrpcInbound>) -> Self {
        Self { default, health }
    }

    /// Pick the inbound that owns `method` — `health` if the path is
    /// under [`HEALTH_SERVICE_PREFIX`], otherwise `default`.
    fn pick(&self, method: &str) -> &Arc<dyn GrpcInbound> {
        if method.starts_with(HEALTH_SERVICE_PREFIX) {
            &self.health
        } else {
            &self.default
        }
    }
}

impl GrpcInbound for MethodPathRouter {
    fn handle_unary(
        &self,
        request: GrpcRequest,
    ) -> BoxFuture<'_, GrpcInboundResult<GrpcResponse>> {
        let inbound = Arc::clone(self.pick(&request.method));
        Box::pin(async move { inbound.handle_unary(request).await })
    }

    fn handle_stream(
        &self,
        method:   String,
        metadata: GrpcMetadata,
        messages: GrpcMessageStream,
    ) -> BoxFuture<'_, GrpcInboundResult<(GrpcMessageStream, GrpcMetadata)>> {
        let inbound = Arc::clone(self.pick(&method));
        Box::pin(async move { inbound.handle_stream(method, metadata, messages).await })
    }

    fn health_check(&self) -> BoxFuture<'_, GrpcInboundResult<GrpcHealthCheck>> {
        // Aggregate is "healthy iff the user dispatcher is healthy" —
        // the `HealthService` itself is always healthy by contract and
        // would mask a sick user dispatcher if we AND-ed both.
        let default = Arc::clone(&self.default);
        Box::pin(async move { default.health_check().await })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;

    /// `GrpcInbound` stub that records whether it was hit and returns a
    /// caller-supplied status string in the response body so the test
    /// can assert which inbound served a given request.
    struct LabelInbound {
        label:   &'static str,
        hit:     Arc<AtomicBool>,
        healthy: bool,
    }

    impl GrpcInbound for LabelInbound {
        fn handle_unary(
            &self,
            _request: GrpcRequest,
        ) -> BoxFuture<'_, GrpcInboundResult<GrpcResponse>> {
            self.hit.store(true, Ordering::SeqCst);
            let label = self.label;
            Box::pin(async move {
                Ok(GrpcResponse {
                    body:     label.as_bytes().to_vec(),
                    metadata: GrpcMetadata::default(),
                })
            })
        }

        fn health_check(
            &self,
        ) -> BoxFuture<'_, GrpcInboundResult<GrpcHealthCheck>> {
            let healthy = self.healthy;
            Box::pin(async move {
                Ok(if healthy {
                    GrpcHealthCheck::healthy()
                } else {
                    GrpcHealthCheck::unhealthy("stub")
                })
            })
        }
    }

    fn pair(default_healthy: bool) -> (Arc<AtomicBool>, Arc<AtomicBool>, MethodPathRouter) {
        let default_hit = Arc::new(AtomicBool::new(false));
        let health_hit  = Arc::new(AtomicBool::new(false));
        let router = MethodPathRouter::new(
            Arc::new(LabelInbound {
                label:   "default",
                hit:     Arc::clone(&default_hit),
                healthy: default_healthy,
            }),
            Arc::new(LabelInbound {
                label:   "health",
                hit:     Arc::clone(&health_hit),
                healthy: true,
            }),
        );
        (default_hit, health_hit, router)
    }

    /// @covers: MethodPathRouter::handle_unary — Health prefix routes to health inbound.
    #[tokio::test]
    async fn test_handle_unary_routes_health_method_to_health_inbound() {
        let (default_hit, health_hit, router) = pair(true);
        let req = GrpcRequest::new(
            "/grpc.health.v1.Health/Check",
            vec![],
            Duration::from_secs(1),
        );
        let resp = router.handle_unary(req).await.expect("ok");
        assert_eq!(resp.body, b"health".to_vec());
        assert!(health_hit.load(Ordering::SeqCst));
        assert!(!default_hit.load(Ordering::SeqCst));
    }

    /// @covers: MethodPathRouter::handle_unary — non-Health method routes to default.
    #[tokio::test]
    async fn test_handle_unary_routes_user_method_to_default_inbound() {
        let (default_hit, health_hit, router) = pair(true);
        let req = GrpcRequest::new(
            "/justembed.EmbedService/Embed",
            vec![],
            Duration::from_secs(1),
        );
        let resp = router.handle_unary(req).await.expect("ok");
        assert_eq!(resp.body, b"default".to_vec());
        assert!(default_hit.load(Ordering::SeqCst));
        assert!(!health_hit.load(Ordering::SeqCst));
    }

    /// @covers: MethodPathRouter::handle_unary — method that merely *contains*
    /// the health prefix substring is NOT misrouted — only paths that start
    /// with the prefix are health calls.
    #[tokio::test]
    async fn test_handle_unary_does_not_misroute_method_containing_health_prefix() {
        let (default_hit, health_hit, router) = pair(true);
        // Notice: the "/grpc.health.v1.Health/" string appears mid-path,
        // not at the start.  The router must treat this as a user RPC.
        let req = GrpcRequest::new(
            "/pkg.Service/grpc.health.v1.Health/SomeMethod",
            vec![],
            Duration::from_secs(1),
        );
        let resp = router.handle_unary(req).await.expect("ok");
        assert_eq!(resp.body, b"default".to_vec());
        assert!(default_hit.load(Ordering::SeqCst));
        assert!(!health_hit.load(Ordering::SeqCst));
    }

    /// @covers: MethodPathRouter::health_check — reports default-inbound health.
    #[tokio::test]
    async fn test_health_check_returns_default_inbound_health_when_default_healthy() {
        let (_, _, router) = pair(true);
        let h = router.health_check().await.expect("ok");
        assert!(h.healthy);
    }

    /// @covers: MethodPathRouter::health_check — flags overall unhealthy when
    /// the default inbound is unhealthy, regardless of the always-healthy
    /// `HealthService`.
    #[tokio::test]
    async fn test_health_check_returns_unhealthy_when_default_inbound_is_unhealthy() {
        let (_, _, router) = pair(false);
        let h = router.health_check().await.expect("ok");
        assert!(!h.healthy);
    }
}
