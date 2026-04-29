//! Integration tests for the standard `grpc.health.v1.Health` service
//! exposed alongside `EmbedService`.
//!
//! Strategy
//! --------
//! These tests exercise the same wire path the production daemon uses:
//!
//!   client → http2 → ingress → MethodPathRouter
//!                                ├─ /grpc.health.v1.Health/* → HealthService
//!                                └─ /justembed.EmbedService/* → dispatcher
//!
//! They verify:
//!   * `Health/Check` for a registered service returns SERVING (1).
//!   * `Health/Check` for an empty service name returns the overall
//!     status (1).
//!   * `Health/Check` for an unknown service returns NotFound (5).
//!   * Routing: a `Check` call does *not* hit the embed dispatcher.
//!   * Routing: an `Embed` call does *not* hit the health service.
//!
//! `EmbedService` is wired with a deterministic stub handler so the
//! tests do not need to load a real GGUF model.

use std::any::Any;
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{BufMut, Bytes, BytesMut};
use edge_domain::{Handler, HandlerError, HandlerRegistry};
use http::{Request, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper_util::rt::{TokioExecutor, TokioIo};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use swe_edge_ingress_grpc::{
    GrpcHandlerAdapter, GrpcInboundError, HandlerRegistryDispatcher, HealthService,
    ServingStatus, TonicGrpcServer, HEALTH_CHECK_METHOD,
};
use swe_embedding_systemd::{
    EmbedRequest, EmbedResponse, EMBED_METHOD_PATH, FloatVector, MethodPathRouter,
};

// ── Stub embed handler — same fixed-vector behaviour as the embed test ───────

const STUB_MODEL_ID: &str  = "stub-embed-model";
const FIXED_VECTOR:  [f32; 2] = [0.5_f32, -0.5_f32];

struct StubEmbedHandler;

#[async_trait]
impl Handler<EmbedRequest, EmbedResponse> for StubEmbedHandler {
    fn id(&self) -> &str { EMBED_METHOD_PATH }
    fn pattern(&self) -> &str { "Embed" }

    async fn execute(&self, req: EmbedRequest) -> Result<EmbedResponse, HandlerError> {
        if req.input.is_empty() {
            return Err(HandlerError::InvalidRequest("input must not be empty".into()));
        }
        Ok(EmbedResponse {
            model:      STUB_MODEL_ID.to_string(),
            embeddings: req
                .input
                .iter()
                .map(|_| FloatVector { values: FIXED_VECTOR.to_vec() })
                .collect(),
        })
    }

    async fn health_check(&self) -> bool { true }
    fn as_any(&self) -> &dyn Any { self }
}

// ── Codec helpers — match the daemon's wire codecs ───────────────────────────

fn decode_embed_request(bytes: &[u8]) -> Result<EmbedRequest, GrpcInboundError> {
    EmbedRequest::decode(bytes).map_err(|e| {
        GrpcInboundError::InvalidArgument(format!("decode EmbedRequest: {e}"))
    })
}

fn encode_embed_response(resp: &EmbedResponse) -> Vec<u8> {
    let mut buf = Vec::with_capacity(resp.encoded_len());
    resp.encode(&mut buf).expect("EmbedResponse encode into Vec<u8> is infallible");
    buf
}

// ── Wire helpers — match `grpc_health_v1_int_test.rs` upstream ───────────────

fn grpc_frame(payload: &[u8]) -> Bytes {
    let mut buf = BytesMut::with_capacity(5 + payload.len());
    buf.put_u8(0); // not compressed
    buf.put_u32(payload.len() as u32);
    buf.put_slice(payload);
    buf.freeze()
}

fn parse_first_grpc_frame(data: &Bytes) -> Option<Bytes> {
    if data.len() < 5 {
        return None;
    }
    let len = u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as usize;
    if data.len() < 5 + len {
        return None;
    }
    Some(data.slice(5..5 + len))
}

/// Encode a `HealthCheckRequest { service }` as proto bytes.
fn encode_health_check_request(service: &str) -> Vec<u8> {
    let bytes = service.as_bytes();
    if bytes.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(2 + bytes.len());
    out.push(0x0a); // tag 1, wire type 2 (length-delimited)
    encode_varint(bytes.len() as u64, &mut out);
    out.extend_from_slice(bytes);
    out
}

fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Decode a `HealthCheckResponse` payload to its `ServingStatus` int value.
///
/// Returns `0` for an empty body (proto3 default), `-1` if the wire
/// shape is unrecognised.
fn decode_status(payload: &[u8]) -> i32 {
    if payload.is_empty() {
        return 0;
    }
    if payload[0] != 0x08 {
        return -1;
    }
    let mut result = 0i64;
    let mut shift  = 0u32;
    for byte in &payload[1..] {
        result |= ((byte & 0x7f) as i64) << shift;
        if byte & 0x80 == 0 {
            return result as i32;
        }
        shift += 7;
    }
    -1
}

// ── Server scaffolding — embed + health behind one router ────────────────────

/// Bring up a server with the embed handler registered + a
/// [`HealthService`] populated with the embed service slot.
async fn start_test_server() -> (SocketAddr, Arc<HealthService>, oneshot::Sender<()>) {
    let registry: Arc<HandlerRegistry<Vec<u8>, Vec<u8>>> =
        Arc::new(HandlerRegistry::new());
    let adapter = GrpcHandlerAdapter::new(
        Arc::new(StubEmbedHandler) as Arc<dyn Handler<EmbedRequest, EmbedResponse>>,
        decode_embed_request,
        encode_embed_response,
    );
    registry.register(Arc::new(adapter));
    let dispatcher = Arc::new(HandlerRegistryDispatcher::new(registry));

    let health = Arc::new(HealthService::new());
    health.set_overall_status(ServingStatus::Serving);
    health.set_status("justembed.EmbedService", ServingStatus::Serving);

    let router = Arc::new(MethodPathRouter::new(
        dispatcher,
        health.clone() as Arc<dyn swe_edge_ingress_grpc::GrpcInbound>,
    ));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr     = listener.local_addr().unwrap();
    let server   = TonicGrpcServer::new("127.0.0.1:0", router)
        .allow_unauthenticated(true);

    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        server
            .serve_with_listener(listener, async move { let _ = rx.await; })
            .await
            .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    (addr, health, tx)
}

async fn grpc_call(
    addr: SocketAddr,
    path: &str,
    payload: &[u8],
) -> (StatusCode, Option<String>, Bytes) {
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let io     = TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http2::Builder::new(TokioExecutor::new())
        .handshake(io)
        .await
        .unwrap();
    tokio::spawn(conn);

    let req = Request::builder()
        .method("POST")
        .uri(format!("http://{addr}{path}"))
        .header("content-type", "application/grpc")
        .header("te", "trailers")
        .header("grpc-timeout", "10S")
        .body(Full::new(grpc_frame(payload)))
        .unwrap();

    let resp        = sender.send_request(req).await.unwrap();
    let status      = resp.status();
    let collected   = resp.into_body().collect().await.unwrap();
    let grpc_status = collected
        .trailers()
        .and_then(|t| t.get("grpc-status"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let data = collected.to_bytes();
    (status, grpc_status, data)
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// @covers: grpc.health.v1.Health.Check — registered service returns SERVING.
///
/// Sanity that the upstream `HealthService` is wired into the embed
/// daemon's gRPC server through the `MethodPathRouter` and that the
/// per-service status is reported correctly.  This is the headline
/// acceptance test for the follow-up PR.
#[tokio::test]
async fn test_health_check_returns_serving_for_registered_embed_service() {
    let (addr, _health, _shutdown) = start_test_server().await;

    let body = encode_health_check_request("justembed.EmbedService");
    let (http_status, grpc_status, data) =
        grpc_call(addr, HEALTH_CHECK_METHOD, &body).await;

    assert_eq!(http_status, StatusCode::OK);
    assert_eq!(grpc_status.as_deref(), Some("0"), "grpc-status must be Ok");

    let payload = parse_first_grpc_frame(&data)
        .expect("Health/Check must produce one response frame");
    // ServingStatus::Serving == 1
    assert_eq!(decode_status(&payload), 1, "expected SERVING (1)");
}

/// @covers: grpc.health.v1.Health.Check — empty service name returns the
/// overall server status.  We set the overall slot to SERVING at
/// startup, so the wire response must be `Serving == 1`.
#[tokio::test]
async fn test_health_check_with_empty_service_returns_overall_serving_status() {
    let (addr, _health, _shutdown) = start_test_server().await;

    let body = encode_health_check_request("");
    let (_, grpc_status, data) = grpc_call(addr, HEALTH_CHECK_METHOD, &body).await;

    assert_eq!(grpc_status.as_deref(), Some("0"));
    let payload = parse_first_grpc_frame(&data).expect("response frame");
    assert_eq!(decode_status(&payload), 1, "expected SERVING (1)");
}

/// @covers: grpc.health.v1.Health.Check — unknown service returns NotFound.
///
/// Per the gRPC health-check spec, querying a service that was never
/// registered with `set_status` must return `tonic::Code::NotFound (5)`.
#[tokio::test]
async fn test_health_check_returns_not_found_for_unregistered_service() {
    let (addr, _health, _shutdown) = start_test_server().await;

    let body = encode_health_check_request("never.registered.Service");
    let (_, grpc_status, _) = grpc_call(addr, HEALTH_CHECK_METHOD, &body).await;
    // tonic::Code::NotFound == 5
    assert_eq!(grpc_status.as_deref(), Some("5"));
}

/// @covers: MethodPathRouter — `Embed` calls reach the embed dispatcher,
/// not the health service.  Without this, a routing bug could silently
/// turn every Embed call into a malformed health response.
#[tokio::test]
async fn test_embed_call_is_routed_to_embed_dispatcher_not_health_service() {
    let (addr, _health, _shutdown) = start_test_server().await;

    let req = EmbedRequest {
        model: "ignored".to_string(),
        input: vec!["hello".to_string()],
    };
    let mut buf = Vec::with_capacity(req.encoded_len());
    req.encode(&mut buf).unwrap();

    let (_, grpc_status, data) = grpc_call(addr, EMBED_METHOD_PATH, &buf).await;
    assert_eq!(grpc_status.as_deref(), Some("0"));

    let payload = parse_first_grpc_frame(&data).expect("response frame");
    let resp = EmbedResponse::decode(payload.as_ref())
        .expect("response must decode as EmbedResponse, not HealthCheckResponse");
    assert_eq!(resp.model, STUB_MODEL_ID, "embed handler must produce the response");
    assert_eq!(resp.embeddings.len(), 1);
}

/// @covers: HealthService::set_status — flipping a registered service to
/// NOT_SERVING is reflected in the next Check call.  Proves the live
/// status table is the source of truth, not a snapshot taken at startup.
#[tokio::test]
async fn test_health_check_reflects_runtime_status_change_for_registered_service() {
    let (addr, health, _shutdown) = start_test_server().await;

    // Flip the service to NOT_SERVING after the server is already running.
    health.set_status("justembed.EmbedService", ServingStatus::NotServing);

    let body = encode_health_check_request("justembed.EmbedService");
    let (_, grpc_status, data) = grpc_call(addr, HEALTH_CHECK_METHOD, &body).await;
    assert_eq!(grpc_status.as_deref(), Some("0"));

    let payload = parse_first_grpc_frame(&data).expect("response frame");
    // ServingStatus::NotServing == 2
    assert_eq!(decode_status(&payload), 2, "expected NOT_SERVING (2)");
}
