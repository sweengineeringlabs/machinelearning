//! `RegistryGrpcInbound` — gRPC inbound bridge.
//!
//! Decodes proto bytes, looks up a typed [`Handler`] in a
//! [`HandlerRegistry`], runs it, and re-encodes the response.
//!
//! ## Why a bridge
//!
//! The edge ingress speaks raw bytes ([`GrpcInbound`]) and the domain
//! speaks proto-typed [`Handler<EmbedRequest, EmbedResponse>`].  The
//! bridge owns the encoding/decoding boundary so neither layer leaks
//! into the other.
//!
//! ## Error mapping
//!
//! `HandlerError::InvalidRequest` → gRPC `InvalidArgument` (carried through
//! `GrpcInboundError::InvalidArgument`).  Other handler errors map to
//! gRPC `Internal` and the on-wire message is sanitised by the ingress.

use std::sync::Arc;

use edge_domain::{HandlerError, HandlerRegistry};
use futures::future::BoxFuture;
use prost::Message;
use swe_edge_ingress_grpc::{
    GrpcHealthCheck, GrpcInbound, GrpcInboundError, GrpcInboundResult,
    GrpcMetadata, GrpcRequest, GrpcResponse,
};

use crate::api::proto::{EmbedRequest, EmbedResponse};

/// Gateway between the byte-level edge ingress and the typed domain registry.
///
/// One bridge instance handles every method served by this daemon — the
/// registry lookup uses `GrpcRequest.method` (the URI path) as the key.
pub struct RegistryGrpcInbound {
    registry: Arc<HandlerRegistry<EmbedRequest, EmbedResponse>>,
}

impl RegistryGrpcInbound {
    /// Build a bridge over the given handler registry.
    pub fn new(registry: Arc<HandlerRegistry<EmbedRequest, EmbedResponse>>) -> Self {
        Self { registry }
    }
}

impl GrpcInbound for RegistryGrpcInbound {
    fn handle_unary(&self, request: GrpcRequest) -> BoxFuture<'_, GrpcInboundResult<GrpcResponse>> {
        let registry = Arc::clone(&self.registry);
        Box::pin(async move {
            let handler = registry.get(&request.method).ok_or_else(|| {
                GrpcInboundError::Unimplemented(format!(
                    "no handler registered for {}",
                    request.method
                ))
            })?;

            let proto_req = EmbedRequest::decode(request.body.as_slice()).map_err(|e| {
                GrpcInboundError::InvalidArgument(format!("decode EmbedRequest: {e}"))
            })?;

            let proto_resp: EmbedResponse = handler
                .execute(proto_req)
                .await
                .map_err(map_handler_error)?;

            let mut buf = Vec::with_capacity(proto_resp.encoded_len());
            // `encode` writes into a `BufMut`-impl `Vec<u8>`; the only way
            // this can fail is OOM, which is treated as Internal.
            proto_resp.encode(&mut buf).map_err(|e| {
                GrpcInboundError::Internal(format!("encode EmbedResponse: {e}"))
            })?;

            Ok(GrpcResponse { body: buf, metadata: GrpcMetadata::default() })
        })
    }

    fn health_check(&self) -> BoxFuture<'_, GrpcInboundResult<GrpcHealthCheck>> {
        Box::pin(async move { Ok(GrpcHealthCheck::healthy()) })
    }
}

/// Map a domain-level [`HandlerError`] onto a gRPC-shaped [`GrpcInboundError`].
fn map_handler_error(e: HandlerError) -> GrpcInboundError {
    match e {
        HandlerError::InvalidRequest(msg) => GrpcInboundError::InvalidArgument(msg),
        HandlerError::Unsupported(msg)    => GrpcInboundError::Unimplemented(msg),
        HandlerError::Unhealthy           => GrpcInboundError::Unavailable("handler unhealthy".into()),
        HandlerError::ExecutionFailed(m)  => GrpcInboundError::Internal(m),
        HandlerError::Other(m)            => GrpcInboundError::Internal(m),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: map_handler_error — InvalidRequest maps to InvalidArgument.
    #[test]
    fn test_map_handler_error_invalid_request_maps_to_invalid_argument() {
        let mapped = map_handler_error(HandlerError::InvalidRequest("bad".into()));
        match mapped {
            GrpcInboundError::InvalidArgument(m) => assert_eq!(m, "bad"),
            other => panic!("unexpected mapping: {other:?}"),
        }
    }

    /// @covers: map_handler_error — ExecutionFailed maps to Internal so its
    /// message is sanitised at the wire.
    #[test]
    fn test_map_handler_error_execution_failed_maps_to_internal() {
        let mapped = map_handler_error(HandlerError::ExecutionFailed("model panic".into()));
        match mapped {
            GrpcInboundError::Internal(m) => assert_eq!(m, "model panic"),
            other => panic!("unexpected mapping: {other:?}"),
        }
    }

    /// @covers: map_handler_error — Unhealthy maps to Unavailable so clients
    /// retry per the gRPC convention rather than treating it as a final error.
    #[test]
    fn test_map_handler_error_unhealthy_maps_to_unavailable() {
        let mapped = map_handler_error(HandlerError::Unhealthy);
        match mapped {
            GrpcInboundError::Unavailable(_) => {}
            other => panic!("unexpected mapping: {other:?}"),
        }
    }
}
