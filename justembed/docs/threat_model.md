# justembed Threat Model

Scope: justembed daemon as deployed via `embed serve`, exposing one
transport — gRPC (`justembed.EmbedService/Embed`) — over a single
loaded GGUF embedding model.

This document captures the threats the daemon defends against, the
controls it relies on, and the residual risks the operator owns.  It
does **not** repeat hardening covered transitively by the upstream edge
ingress; assertions about the wire surface refer to the contracts the
edge crates already verify in their own test suites.

---

## Trust boundaries

| Boundary | Side A (untrusted) | Side B (trusted) | Crossed by |
|---------|--------------------|------------------|------------|
| External client → daemon | gRPC client | daemon process | TCP socket |
| Daemon → filesystem | daemon process | GGUF model file, TLS PEMs, application.toml | startup load + read |
| Operator → config | operator | application.toml in XDG dirs | process restart |

The daemon is single-tenant: one model, one process, one config.  There
is no per-request cryptographic identity for embeddings — the vectors
are purely a function of the input text plus the loaded model.

---

## Assets

1. **Model availability** — the embedding service must remain responsive
   for downstream consumers (swerag, llmboot, etc.).  A wedged or
   crashed daemon stalls retrieval pipelines.
2. **Model integrity** — the GGUF file on disk; tampering would silently
   change every embedding the cluster produces.
3. **Input confidentiality (in-flight)** — embedded text often contains
   user content; on-the-wire interception leaks it.
4. **TLS private key** — disclosure breaks transport confidentiality
   and, with mTLS, lets an attacker impersonate the server.
5. **Process isolation** — the daemon must not be exploitable into
   arbitrary code execution from network input.

Out of scope: model-extraction attacks (learning the weights via crafted
queries), embedding-inversion attacks (recovering text from vectors),
and side-channel attacks on the host.  These are research-grade
adversaries; the daemon is not designed to defeat them.

---

## Threats and controls

### T1 — Denial of service via oversized request

* **Attacker capability:** any client that can reach the gRPC port.
* **Impact:** memory exhaustion, GC pressure, model thread starvation.
* **Control:** gRPC ingress applies `max_message_bytes` (default 4 MiB) at the
  framing layer; over-limit messages return `ResourceExhausted` (status 8)
  before the proto decoder runs.
* **Verification:** edge ingress test
  `test_server_enforces_message_size_limit_with_resource_exhausted`.

### T2 — Denial of service via slow / never-completing request

* **Attacker capability:** open a connection and never finish writing,
  or ask for an embedding of a pathological input.
* **Impact:** request handler tied up indefinitely, blocking other work.
* **Control:** every gRPC request carries a mandatory deadline parsed
  from the `grpc-timeout` header (default 30 s when absent).  The
  ingress races the handler future against `tokio::time::sleep(deadline)`
  and aborts past-deadline calls with `DeadlineExceeded` (status 4).
* **Verification:** edge ingress tests
  `test_server_returns_deadline_exceeded_when_handler_overruns_deadline`
  and `test_server_rejects_past_deadline_request_before_handler_dispatch`.

### T3 — Denial of service via client disconnect mid-call

* **Attacker capability:** open many connections, send a request, drop
  the TCP stream before reading the response.
* **Impact:** background work continues for an unobserved client,
  pinning a CPU.
* **Control:** the gRPC ingress fires a `CancellationToken` when the
  HTTP/2 stream closes and aborts the handler future.
* **Verification:** edge ingress test
  `test_server_returns_cancelled_when_client_drops_connection_mid_call`.

### T4 — Information disclosure via error messages

* **Attacker capability:** malformed or hostile request input.
* **Impact:** stack traces, file paths, identity strings leak through
  error responses.
* **Control:** `HandlerError::ExecutionFailed` and `Other` map to
  `GrpcInboundError::Internal`; the wire `grpc-message` is replaced
  with a fixed sanitised constant by the edge status mapper.  The
  raw message is logged server-side at WARN.
* **Verification:** edge ingress test
  `test_server_sanitizes_internal_error_message_on_wire`.

### T5 — Confidentiality on the wire

* **Attacker capability:** on-path observer between client and daemon.
* **Impact:** disclosure of input text and embedding output.
* **Control:** TLS-by-default for production deployments.
  `[embedding.grpc.tls]` accepts `cert_pem_path` / `key_pem_path` and an
  optional `client_ca_pem_path` to enable mTLS.  Cert / key load
  failures surface at startup, not at first connection.
* **Default posture:** the bundled `application.toml` ships with no TLS
  block and a 127.0.0.1 bind, i.e. development mode.  Production
  operators MUST set TLS paths and bind to an internal interface.

### T6 — Authentication / authorisation

* **Attacker capability:** any reachable network identity.
* **Impact:** unauthorised consumption of model time, content
  exfiltration if input text is sensitive, side-stepping per-tenant
  quotas.
* **Control:** the gRPC ingress supports a pluggable inbound trait
  (`GrpcInbound`) which can be wrapped with an auth decorator.  The
  daemon today does NOT register such a decorator and therefore opens
  with `allow_unauthenticated = true`.  Setting it to `false` is a
  startup-time error until an interceptor is wired.
* **Residual risk:** without a decorator, every request is implicitly
  authorised.  Until that gap closes, operators should rely on network
  policy (mTLS client cert, VPC/L3 firewall) for AuthN/AuthZ.

### T7 — Malicious input causing model crash

* **Attacker capability:** craft an input string that triggers a
  tokeniser or model-forward panic.
* **Impact:** process crash and downtime; potential out-of-bounds
  access in `unsafe` model paths.
* **Control:** tokenisation runs first in `embed_inputs` and surfaces
  errors as `EmbedError::Tokenization` → `InvalidArgument` rather than
  panicking.  Model errors (`EmbedError::Embed`) are caught and
  surfaced as `Internal`, sanitised on the gRPC wire.  The model crate
  is built with `unsafe_code = "deny"` lint where feasible.
* **Residual risk:** llama.cpp-backed paths (when enabled) call into
  C and are outside this lint guarantee — keep them off the production
  feature flag unless explicitly required.

### T8 — Tampering with the GGUF model file

* **Attacker capability:** local filesystem access.
* **Impact:** silent corruption of every embedding the cluster
  generates; downstream retrieval results poisoned.
* **Control:** none in-process — the daemon parses what's on disk.
* **Required operator practice:** put the GGUF on a read-only mount
  owned by a service user the daemon does not own; pin a known SHA256
  in the deployment pipeline; alert on file mtime change.

### T9 — Configuration tampering

* **Attacker capability:** write access to `application.toml`.
* **Impact:** swap GGUF path, disable TLS, lower message size cap,
  rebind to 0.0.0.0.
* **Control:** layered XDG load — site-wide config under
  `$XDG_CONFIG_DIRS/llminference/application.toml` overrides bundled
  defaults; user override under `$XDG_CONFIG_HOME` overrides site.
* **Required operator practice:** treat `application.toml` as
  deployment-managed, owned root, mode 0644 at most.  Never let the
  service user own it.

---

## Defaults summary

| Knob | Default | Production setting |
|------|---------|-------------------|
| gRPC bind | `127.0.0.1:8181` | internal interface only |
| `max_message_bytes` | 4 MiB | tighten per workload |
| `allow_unauthenticated` | `true` | `false` once auth interceptor lands |
| TLS | none | mandatory |
| mTLS | none | recommended |
| Default deadline | 30 s | tune per workload |

Every default is biased to "developer can run this on a laptop without
ceremony"; every production deployment must walk this table and harden.
