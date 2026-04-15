# llmserv Daemon Code Review

**Date:** 2026-04-15
**Scope:** `swellmd` (HTTP daemon at `main/features/daemon/main`)
**Verdict:** **Builds clean. Functional. Honest concerns around `usage` accounting, unbounded generation, and warning hygiene.**

This review was verified by reading the cited files and running `cargo build -p swellmd` from `llmserv/`. It compiled in 3m 09s with warnings (see §5).

---

## What's solid

- **Boundary validation is real.** `core/router.rs:63-97` rejects negative temperature, `top_k == 0`, `top_p` outside `(0, 1]`, and non-positive `repetition_penalty` with explicit `InvalidRequest` errors that name the offending value.
- **Content-correctness regression tests exist.** `tests/content_correctness.rs` sends three fixed prompts ("capital of France", "13 × 17", "largest country in South America") and asserts at least one expected substring appears in the response, lowercased on both sides. Gated by env vars + `--ignored` so they only run when a real model is loaded. Documented thoroughly in the file's module doc. This is the single best piece of test code in the project and the pattern other features should adopt.
- **Throttling is config-driven, not string-dispatched.** Permits acquired via `state.throttle.try_acquire()` (`router.rs:117-119`); capacity reported back to the client on `AtCapacity`.
- **DI for backends is explicit.** Backend registry built at startup; the request handler never branches on a backend name string.

---

## Status update — 2026-04-15 fix pass

The three top-priority concerns below have been addressed in code. Build verified clean (`cargo build -p swellmd` finished in 1m08s; 6/6 lib unit tests pass; no new warnings introduced). Test-coverage gap acknowledged at the bottom of this section.

| # | Concern | Status | Files touched |
|---|---|---|---|
| 1 | Whitespace token counts | **Fixed** — `count_tokens()` helper calls real tokenizer at all three sites | `core/router.rs` |
| 2 | Streaming `usage` missing | **Fixed** — channel item type is now `StreamItem::{Token, Done(Usage)}`; final chunk carries `usage` and `finish_reason: "stop"` before `[DONE]` | `core/router.rs`, `api/types.rs` |
| 3 | Unbounded generation / permit leak | **Fixed** — `[generation].request_timeout_secs` config wired into `AppState.request_timeout` and applied via `build_deadline()` at all three sites; default `0` preserves prior behavior | `api/config.rs`, `core/state.rs`, `bin/serve.rs`, `core/router.rs`, `config/application.toml` |

**Test-coverage gap (honest):** no automated regression test yet asserts (a) that `usage.completion_tokens` differs from `split_whitespace().count()`, (b) that the streaming `usage` chunk is actually emitted, or (c) that a deadline truncates generation. `content_correctness.rs` exercises `Generator` directly, not the HTTP layer, so a regression that re-introduced `split_whitespace()` in the handler would still pass CI. Fixing this requires either a mock `Tokenizer`+`Model` stack for unit tests or an HTTP integration test gated alongside `LLMSERV_CC_*`. Not done in this pass.

---

## Original verified concerns (kept for history)

### 1. Token counts in `usage` are fake
`router.rs:154-158` (chat blocking) and `:321-322` (legacy completions):

```rust
let prompt_tokens = owned_messages
    .iter()
    .map(|(_, c)| c.split_whitespace().count())
    .sum::<usize>();
let completion_tokens = output.split_whitespace().count();
```

Tokenizers are subword. `"don't"` is 2 tokens; `"unbelievable"` may be 3-4. Whitespace counts are systematically wrong — and the sign of the error depends on the language and the tokenizer. The `Usage` block returned to the client is a confident lie. No test asserts that reported counts match what the tokenizer actually produced.

**Fix:** call `state.model.tokenizer().encode(...)` on the rendered prompt and the completion string. Store the resulting lengths.

### 2. Streaming responses report no usage at all
`handle_streaming` (`router.rs:195-269`) emits `chat.completion.chunk` events, then a final `[DONE]` sentinel. It never assembles a `Usage` block. Clients that depend on the OpenAI streaming convention (`finish_reason` chunk carrying `usage`) will see nothing.

**Fix:** count tokens as they pass through the `complete_turn_stream` callback (`:235-240`), then emit a final chunk with `finish_reason: "stop"` and the usage payload before `[DONE]`.

### 3. Generation is unbounded; throttle slot can leak
`CompletionParams { ..., deadline: None }` at three sites (`router.rs:140`, `:226`, `:311`). The blocking task holds the semaphore permit (`let _permit = permit;` at `:132`, `:218`, `:303`) for the entire forward pass. If a model hangs — bad input, broken kv-cache state, OOM thrash — the permit is held until process restart. Capacity silently degrades.

**Fix:** wire a configurable `request_timeout` from `core/config.rs` into a `Some(Instant::now() + dur)` deadline. Treat deadline expiry as a recoverable `GenerationFailed`, drop the permit.

### 4. Warning hygiene
`cargo build -p swellmd` produces:
- 54 warnings in `swe-ml-tensor`
- 8 warnings in `rustml-model`
- 1 warning in `rustml-arch-llama` (unused `pos_embedding` at `core/builder.rs:61`)

Per the global metaprompt §2: production builds should have zero warnings or explicitly justified ones. None of these are justified inline.

**Fix:** triage the `swe-ml-tensor` and `rustml-model` warnings — most are likely dead code or `_`-prefix candidates. Keep the build clean so genuinely new warnings stand out.

---

## Not concerns (corrections to earlier review pass)

- **`collect_messages` does not panic on malformed input.** It clones owned `String`s out of already-deserialized `ChatMessage`s. Serde would have rejected a non-string `content` field upstream. An empty `role` produces an empty filter result, not a panic.

---

## Concerns to address, ordered by risk

| # | Issue | File:line | Severity |
|---|---|---|---|
| 1 | `usage.prompt_tokens` / `completion_tokens` use whitespace splitting | `router.rs:154-158`, `:321-322` | High |
| 2 | Streaming response never emits a `usage` block | `router.rs:195-269` | High |
| 3 | `deadline: None` + permit-held-across-blocking can leak throttle slots | `router.rs:140,226,311` | High |
| 4 | 63 build warnings across `swe-ml-tensor`, `rustml-model`, `rustml-arch-llama` | (multiple) | Medium |
| 5 | `chat_template` loaded as opaque string; no test that it renders cleanly per-model | `core/loader.rs` | Medium (this is the class of bug that produced P9 gemma3 garbage) |

---

## Recommended next actions

1. Replace whitespace token counts with real tokenizer calls in both blocking and streaming paths; add an assertion in `tests/content_correctness.rs` that `usage.completion_tokens > 0` matches `tokenizer.encode(response).len()`.
2. Add `request_timeout: Duration` to daemon config; thread into `CompletionParams.deadline`.
3. Burn down the 63 build warnings or document each remaining one with a `#[allow(...)]` and a one-line reason. Then add `-D warnings` to CI for these crates.
4. Add a chat-template smoke test: for each registered model, render the template against a fixed `[(user, "hi")]` and snapshot the result. Catches template breakage before content-correctness does.
