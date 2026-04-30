# justembed Development Backlog

Items are ordered by risk/priority within each phase. Completed items are marked `[x]`.

---

## Phase 1 — Security hardening

- [ ] **Wire auth interceptor (T6)** — `allow_unauthenticated = true` is the current default. Implement a `GrpcInbound` auth decorator (bearer token or mTLS client cert validation) and wire it in `start_grpc_server`. Until this lands, every reachable network identity has implicit authorisation. Track the residual risk in `../3-design/threat_model.md` § T6.
- [ ] **Admission control** — no concurrent-request cap today. `tokio::task::spawn_blocking` pool is unbounded. If embedding throughput becomes a bottleneck under burst load, add a `SemaphoreThrottle` wrapper at the `Handler` level following the `swellmd` pattern (not inside `embed_inputs`).
- [ ] **GGUF SHA256 pinning tooling** — T8 requires operators to pin the model file hash. Add a `embed verify-model` subcommand that prints the SHA256 of the configured GGUF so it can be baked into deployment pipelines.

---

## Phase 2 — Observability

- [ ] **Benchmark baseline** — no throughput numbers exist yet. Run the benchmark suite defined in `../7-operations/benchmarks/baseline.md` and fill in the results table.
- [ ] **Structured logging** — current logs are unstructured strings via `log`/`env_logger`. Add request-scoped fields (model id, input count, token count, latency ms) so logs are queryable.
- [ ] **Metrics endpoint** — expose Prometheus-compatible counters (request count, error count by code, embedding latency histogram, token count) via a sidecar HTTP endpoint or gRPC reflection.

---

## Phase 3 — Test coverage

- [ ] **Manual test checklist** — no `../5-testing/` exists. Write a manual test doc covering: no-model boot, model load, single embed, batch embed, oversized request, empty input, TLS handshake, health Check + Watch. Follow the pattern in `llmforge/docs/testing/manual_testing.md`.
- [ ] **Error-path regression tests** — integration tests currently cover the happy path and the no-model mode. Add regression tests for: `EmptyInput`, `Tokenization` error → `InvalidArgument`, `Embed` error → `Internal` (sanitized), oversized message → `ResourceExhausted`.
- [ ] **Content-correctness test** — no test verifies that `embed_inputs` returns numerically correct vectors for a known input. Add a golden-output test: fixed prompt → tokenize → embed → compare against a pre-computed reference vector (within float tolerance). A test that only checks "didn't crash" is not sufficient.

---

## Phase 4 — Feature extensions

- [ ] **Batch size > 1** — `embed_inputs` currently embeds inputs sequentially (one `spawn_blocking` call per vector). Add a batched forward path (`PoolingStrategy::Mean` across a stacked tensor) once the model crate supports it.
- [ ] **Model metadata endpoint** — expose model id, embedding dimension, and vocab size via a `justembed.EmbedService/ModelInfo` unary RPC (parallel to `swellmd`'s `/v1/models`). Useful for callers that need to know the output dimension before allocating.
- [ ] **Graceful shutdown drain** — current `SIGINT` handler kills in-flight requests. Add a drain window (configurable, default 5 s) that stops accepting new connections but waits for running handlers to complete.
