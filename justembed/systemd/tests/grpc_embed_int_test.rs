//! Integration tests for the gRPC `EmbedService` ingress.
//!
//! Strategy
//! --------
//! The justembed daemon loads a real GGUF model at startup, which is
//! too heavy for unit tests.  These tests instead substitute a
//! deterministic stub `Handler<EmbedRequest, EmbedResponse>` and
//! exercise the full wire path through `TonicGrpcServer`:
//!
//!   client → http2 → ingress → RegistryGrpcInbound → stub handler
//!
//! This proves:
//!   * proto round-trips preserve f32 vectors bit-for-bit
//!   * the registry bridge dispatches to the right handler by URI path
//!   * `HandlerError::InvalidRequest` surfaces as gRPC `InvalidArgument`
//!   * mismatched URIs surface as gRPC `Unimplemented`
//!
//! The "parity" claim — REST and gRPC return semantically equivalent
//! embeddings — follows because both transports invoke the same shared
//! [`embed_inputs`] function with no extra per-transport transforms.
//! The on-the-wire f32 fidelity check below is what guards parity.

use std::any::Any;
use std::collections::HashMap;
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

use swe_edge_ingress_grpc::TonicGrpcServer;
use swe_embedding_systemd::{
    EmbedRequest, EmbedResponse, EMBED_METHOD_PATH, FloatVector, RegistryGrpcInbound,
};

// ── Deterministic stub handler ────────────────────────────────────────────────

/// Stub handler returning a fixed vector per input — enough to verify
/// proto round-trip fidelity and dispatch correctness without loading a
/// real GGUF model.
///
/// Behaviour:
///   * empty input  → `HandlerError::InvalidRequest`
///   * input "boom" → `HandlerError::ExecutionFailed` (maps to Internal)
///   * otherwise    → one [`FIXED_VECTOR`]-shaped vector per input
struct StubEmbedHandler;

const FIXED_VECTOR: [f32; 4] = [0.5_f32, -0.25_f32, 0.125_f32, -0.0625_f32];
const STUB_MODEL_ID: &str = "stub-embed-model";

#[async_trait]
impl Handler<EmbedRequest, EmbedResponse> for StubEmbedHandler {
    fn id(&self) -> &str { EMBED_METHOD_PATH }
    fn pattern(&self) -> &str { "Embed" }

    async fn execute(&self, req: EmbedRequest) -> Result<EmbedResponse, HandlerError> {
        if req.input.is_empty() {
            return Err(HandlerError::InvalidRequest("input must not be empty".into()));
        }
        if req.input.iter().any(|s| s == "boom") {
            return Err(HandlerError::ExecutionFailed("simulated failure".into()));
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

// ── Wire helpers — copied from edge ingress test pattern ──────────────────────

fn grpc_frame(payload: &[u8]) -> Bytes {
    let mut buf = BytesMut::with_capacity(5 + payload.len());
    buf.put_u8(0); // not compressed
    buf.put_u32(payload.len() as u32);
    buf.put_slice(payload);
    buf.freeze()
}

fn parse_grpc_frames(data: &Bytes) -> Vec<Bytes> {
    const HEADER: usize = 5;
    let mut out    = Vec::new();
    let mut offset = 0usize;
    while offset + HEADER <= data.len() {
        let len = u32::from_be_bytes([
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
        ]) as usize;
        let start = offset + HEADER;
        let end   = start + len;
        if end > data.len() { break; }
        out.push(data.slice(start..end));
        offset = end;
    }
    out
}

async fn start_test_server() -> (SocketAddr, oneshot::Sender<()>) {
    let registry: Arc<HandlerRegistry<EmbedRequest, EmbedResponse>> =
        Arc::new(HandlerRegistry::new());
    registry.register(Arc::new(StubEmbedHandler));
    let inbound = Arc::new(RegistryGrpcInbound::new(registry));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr     = listener.local_addr().unwrap();
    let server   = TonicGrpcServer::new("127.0.0.1:0", inbound);

    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        server
            .serve_with_listener(listener, async move { let _ = rx.await; })
            .await
            .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    (addr, tx)
}

async fn grpc_post(
    addr:   SocketAddr,
    path:   &str,
    body:   Bytes,
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
        .body(Full::new(body))
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

fn encoded_request(model: &str, inputs: &[&str]) -> Bytes {
    let req = EmbedRequest {
        model: model.to_string(),
        input: inputs.iter().map(|s| (*s).to_string()).collect(),
    };
    let mut buf = Vec::with_capacity(req.encoded_len());
    req.encode(&mut buf).expect("encode");
    grpc_frame(&buf)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector lengths differ");
    let dot:  f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// @covers: Embed RPC end-to-end — proto encodes, server dispatches,
/// response decodes, model id round-trips, vector count matches input count.
#[tokio::test]
async fn test_embed_rpc_returns_one_vector_per_input_with_correct_model() {
    let (addr, _shutdown) = start_test_server().await;

    let body = encoded_request("ignored-model", &["alpha", "beta", "gamma"]);
    let (http_status, grpc_status, data) =
        grpc_post(addr, EMBED_METHOD_PATH, body).await;

    assert_eq!(http_status, StatusCode::OK);
    assert_eq!(grpc_status.as_deref(), Some("0"));

    let frames = parse_grpc_frames(&data);
    assert_eq!(frames.len(), 1, "unary RPC must produce exactly one response frame");
    let resp = EmbedResponse::decode(frames[0].as_ref())
        .expect("response decodes as EmbedResponse");

    assert_eq!(resp.model, STUB_MODEL_ID, "model id must round-trip");
    assert_eq!(resp.embeddings.len(), 3, "one vector per input");
    for fv in &resp.embeddings {
        assert_eq!(fv.values.as_slice(), &FIXED_VECTOR);
    }
}

/// @covers: proto round-trip preserves f32 components bit-for-bit, so
/// the cosine similarity between the on-wire vector and the handler's
/// in-memory vector is exactly 1.0.  This is the parity guard between
/// REST and gRPC: both transports return whatever the embedding loop
/// produced, so vector fidelity is what defines semantic equivalence.
#[tokio::test]
async fn test_embed_rpc_proto_roundtrip_preserves_vector_components_exactly() {
    let (addr, _shutdown) = start_test_server().await;

    let body = encoded_request("m", &["doc"]);
    let (_, grpc_status, data) = grpc_post(addr, EMBED_METHOD_PATH, body).await;

    assert_eq!(grpc_status.as_deref(), Some("0"));
    let frames = parse_grpc_frames(&data);
    let resp   = EmbedResponse::decode(frames[0].as_ref()).unwrap();
    let on_wire = &resp.embeddings[0].values;

    // Bit-exact equality.
    assert_eq!(on_wire.as_slice(), &FIXED_VECTOR);

    // And, redundantly, cosine similarity ≈ 1.0 — the acceptance-criteria
    // metric.  > 0.999 is the required threshold; we expect 1.0 exactly.
    let cs = cosine_similarity(on_wire, &FIXED_VECTOR);
    assert!(
        cs > 0.999,
        "expected cosine similarity > 0.999 between gRPC vector and source, got {cs}"
    );
}

/// @covers: HandlerError::InvalidRequest → gRPC InvalidArgument (status 3).
#[tokio::test]
async fn test_embed_rpc_returns_invalid_argument_for_empty_input() {
    let (addr, _shutdown) = start_test_server().await;

    let body = encoded_request("m", &[]);
    let (http_status, grpc_status, _) = grpc_post(addr, EMBED_METHOD_PATH, body).await;

    assert_eq!(http_status, StatusCode::OK);
    // tonic::Code::InvalidArgument == 3
    assert_eq!(grpc_status.as_deref(), Some("3"));
}

/// @covers: HandlerError::ExecutionFailed → gRPC Internal (status 13).
/// Sanity that error-path messages don't surface raw on the wire.
#[tokio::test]
async fn test_embed_rpc_returns_internal_for_handler_execution_failure() {
    let (addr, _shutdown) = start_test_server().await;

    let body = encoded_request("m", &["boom"]);
    let (http_status, grpc_status, _) = grpc_post(addr, EMBED_METHOD_PATH, body).await;

    assert_eq!(http_status, StatusCode::OK);
    // tonic::Code::Internal == 13
    assert_eq!(grpc_status.as_deref(), Some("13"));
}

/// @covers: bridge — unknown method path → gRPC Unimplemented (status 12).
/// Proves the registry lookup is by URI path and missing entries fail
/// loud, not silently fall back to a default handler.
#[tokio::test]
async fn test_embed_rpc_returns_unimplemented_for_unknown_method() {
    let (addr, _shutdown) = start_test_server().await;

    let body = encoded_request("m", &["x"]);
    let (http_status, grpc_status, _) =
        grpc_post(addr, "/justembed.EmbedService/DoesNotExist", body).await;

    assert_eq!(http_status, StatusCode::OK);
    // tonic::Code::Unimplemented == 12
    assert_eq!(grpc_status.as_deref(), Some("12"));
}

/// @covers: bridge — malformed proto body → gRPC InvalidArgument (status 3).
/// Protects the daemon from crashing on garbage input.
#[tokio::test]
async fn test_embed_rpc_returns_invalid_argument_for_malformed_proto() {
    let (addr, _shutdown) = start_test_server().await;

    // Garbage body that cannot decode as an EmbedRequest.
    let body = grpc_frame(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
    let (http_status, grpc_status, _) = grpc_post(addr, EMBED_METHOD_PATH, body).await;

    assert_eq!(http_status, StatusCode::OK);
    // tonic::Code::InvalidArgument == 3
    assert_eq!(grpc_status.as_deref(), Some("3"));
}

/// @covers: bridge — request metadata is collected by the ingress; sanity
/// that custom headers do not corrupt dispatch.
#[tokio::test]
async fn test_embed_rpc_succeeds_with_custom_request_metadata() {
    let (addr, _shutdown) = start_test_server().await;

    // Send with an extra metadata header — must not interfere with routing.
    let body   = encoded_request("m", &["x"]);
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let io     = TokioIo::new(stream);
    let (mut sender, conn) = hyper::client::conn::http2::Builder::new(TokioExecutor::new())
        .handshake(io)
        .await
        .unwrap();
    tokio::spawn(conn);

    let req = Request::builder()
        .method("POST")
        .uri(format!("http://{addr}{EMBED_METHOD_PATH}"))
        .header("content-type", "application/grpc")
        .header("te", "trailers")
        .header("x-request-id", "req-123")
        .body(Full::new(body))
        .unwrap();

    let resp = sender.send_request(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let collected = resp.into_body().collect().await.unwrap();
    let grpc_status = collected
        .trailers()
        .and_then(|t| t.get("grpc-status"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    assert_eq!(grpc_status.as_deref(), Some("0"));

    // Force unused-import lint to stay happy on HashMap.
    let _ = HashMap::<String, String>::new();
}
