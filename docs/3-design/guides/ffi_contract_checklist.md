# FFI Contract Checklist

**Purpose**: a reusable checklist for reviewing or designing any Rust
`cdylib` (C-ABI shared object) in this repo.

The C ABI only specifies the mechanics of a function call (registers,
stack, struct layout). Every safety property beyond that must be
provided by the library itself — through Rust code, typed wrappers, or
prose documentation. This document captures the properties that any
mature FFI library should define a position on.

Use it for:

- **Design review.** New `cdylib` crate? Walk down the checklist. Every
  row should have a chosen position: *enforced*, *deferred with
  rationale*, or *not applicable with rationale*. No skipped rows.
- **Implementation review.** Existing crate? Walk down to find rows that
  are currently caller-discipline and could be upgraded to
  runtime-enforced.
- **Release readiness.** Before a tagged cdylib release, verify the
  matrix is filled for that version.

The checklist is ordered roughly from "enforced by Rust/the ABI" to
"documented caller discipline."

Current applicable consumer: `llmserv/main/features/ffi/`
(`llmserv.dll` / `libllmserv.so` / `libllmserv.dylib`). Each row cites
how llmserv-ffi addresses it — use as a worked example.

---

## Enforcement levels

Every row below lands at one of:

- **ABI** — the C ABI guarantees this at the instruction level. Nothing
  to do on our side. (Very few things qualify.)
- **Structural** — impossible to violate because the code won't compile
  or the type system prevents it. Strongest.
- **Runtime** — enforced by explicit checks in our function bodies (null
  checks, `catch_unwind`, locks). The misuse is detectable and
  converted to an error code.
- **Documentation** — the rule exists only in the README / header
  comments. Caller discipline. Weakest.

Aim to push rows upward: every documentation row is a design debt that
could become a runtime or structural guarantee.

---

## Checklist

### 1. Calling convention

| # | Property | Enforcement | Verification | llmserv-ffi status |
|---|---|---|---|---|
| 1.1 | Public functions declared `extern "C"` (or `unsafe extern "C"`) | Structural (won't compile) | grep for `pub fn` without `extern "C"` in `src/lib.rs` | Done |
| 1.2 | Public functions marked `#[no_mangle]` so symbols match the header | Structural | grep for `extern "C"` without `#[no_mangle]` or `#[unsafe(no_mangle)]` | Done |
| 1.3 | All publicly visible types in signatures are `#[repr(C)]` | Structural | grep `#[repr(C)]` on enums/structs appearing in signatures | Done (LlmError, LlmTokenCallback) |
| 1.4 | Header regenerated from source; caller doesn't need a Rust toolchain | Runtime (build.rs) | `include/*.h` committed; `git diff --exit-code` in CI | Done (cbindgen via build.rs) |

### 2. Panic safety

| # | Property | Enforcement | Verification | llmserv-ffi status |
|---|---|---|---|---|
| 2.1 | No Rust panic unwinds across the FFI boundary (UB otherwise) | Runtime (`std::panic::catch_unwind` in every extern fn) | grep for `extern "C" fn` body without `catch_unwind`/`wrap()` | Done (every fn calls `wrap()`) |
| 2.2 | Callbacks from caller can panic; library stays consistent | Runtime (`catch_unwind` around each callback invocation) | unit/smoke test that passes a panicking callback | Done (inside `llmserv_complete_stream`) |
| 2.3 | Caught panics become a defined error code, not silent success | Runtime | test: panic inside extern fn returns `LlmError::Panic` | Done (`LlmError::Panic = 4`) |

### 3. Null pointers and invalid strings

| # | Property | Enforcement | Verification | llmserv-ffi status |
|---|---|---|---|---|
| 3.1 | Every pointer argument checked for null before dereference | Runtime (null check → `InvalidInput`) | test: pass NULL, expect error not segfault | Done (via `cstr_to_str` + `.as_ref()`) |
| 3.2 | C-string → Rust `&str` checks UTF-8 validity | Runtime | test: pass invalid UTF-8 bytes | Done (via `CStr::to_str`) |
| 3.3 | Out-pointers (the `*mut *mut T` that writes results) checked for null | Runtime | test: pass NULL for out-pointer | Partial: init checks, others assume valid |

### 4. Memory ownership and allocator boundaries

| # | Property | Enforcement | Verification | llmserv-ffi status |
|---|---|---|---|---|
| 4.1 | Every heap allocation returned across FFI has a matching typed `free_*` function | Documentation + naming | spec: one `free_X` per allocation type | Done (`free_string`, `free_floats`, `free_u32s`) |
| 4.2 | Rust allocates, Rust frees. Never mix with caller's `free()` / `PyMem_Free` / `delete` | Documentation | README "Memory ownership" section | Done |
| 4.3 | Stack-scoped buffers for streaming callbacks (no heap crosses the boundary) | Runtime (CString dropped after callback returns) | read the implementation | Done for `complete_stream` |
| 4.4 | `free_*` accepts null as a no-op | Runtime | test: `free_string(NULL)` | Done |
| 4.5 | Double-free protection | Runtime ideally; else documentation | test: calling `free_string` twice | Documentation (standard C pattern) |
| 4.6 | Debug-mode canary / magic-value allocator guard | Runtime (optional) | — | Not implemented |

### 5. Handle lifecycle

| # | Property | Enforcement | Verification | llmserv-ffi status |
|---|---|---|---|---|
| 5.1 | `init` returns a fully-owned opaque handle; caller never sees fields | Structural (opaque struct declared in header, hidden body) | grep: handle body is private | Done |
| 5.2 | `destroy` is idempotent (second call is a no-op) | Runtime (`guard.take()` after lock) | test: double-destroy doesn't crash | Done |
| 5.3 | Use-after-destroy returns a defined error, not UB | Runtime (leak outer struct, `Option` for inner, return `Destroyed`) | test: call after destroy expects `Destroyed` | Done |
| 5.4 | Destroy concurrent with in-flight call waits for it to finish | Runtime (`RwLock` write waits for readers) | test: spawn thread calling slow op, destroy in main, time the wait | Done (verified ~4s wait) |
| 5.5 | Handle pointer remains valid for process lifetime (cost: bounded leak) | Structural (we deliberately don't free the outer Box) | grep for `Box::from_raw` in `destroy` — should be absent | Done |

### 6. Thread safety

| # | Property | Enforcement | Verification | llmserv-ffi status |
|---|---|---|---|---|
| 6.1 | All types behind the handle are `Send + Sync` (no interior mutability) | Structural (Rust trait bounds) | grep for `RefCell`/`Cell`/`UnsafeCell` in session types | Done (`Model: Send + Sync` bound) |
| 6.2 | Concurrent read operations on the same handle produce consistent results | Runtime (`RwLock::read` + read-only types) | concurrent smoke test: N threads × M calls, check consistency | Done (4000 calls × 16 threads, 0 mismatches) |
| 6.3 | Document whether callbacks can be invoked from caller's thread vs a library thread | Documentation | header / README | Partial: current callbacks run inline |
| 6.4 | Warn callers that concurrency does not scale (rayon contention, etc.) | Documentation | README "Allowed but not useful" section | Done |

### 7. Error reporting

| # | Property | Enforcement | Verification | llmserv-ffi status |
|---|---|---|---|---|
| 7.1 | Every function returns an error code; zero = success | Structural (function signatures) | grep: every `extern fn` returns `c_int` / error enum | Done |
| 7.2 | Error enum exposed in the header with named variants | Structural + cbindgen config | check header has the enum, not opaque `c_int` | Done (LlmError in header) |
| 7.3 | Error variants stable pre-1.0 — new ones only appended at the end | Documentation + review | CI check: diff enum order against baseline | Informal today |
| 7.4 | Optional last-error-string accessor for debug context | Runtime (thread-local buffer) | test | Not implemented (logged via `log`) |

### 8. Callback safety (if any extern fn takes a function pointer)

| # | Property | Enforcement | Verification | llmserv-ffi status |
|---|---|---|---|---|
| 8.1 | Callback type declared as plain function pointer (not `Option<fn>`, which cbindgen mis-renders) | Structural | check `cbindgen`-generated header for plain `typedef` | Done (`LlmTokenCallback`) |
| 8.2 | `user_data` pointer round-trips unchanged (opaque context) | Runtime | test: pass pointer, compare what callback receives | Done (verified `0x29d37484c98` round-trip) |
| 8.3 | Callback panics caught, converted to "stop" signal, not propagated | Runtime (`catch_unwind` around the callback invocation) | test: panicking callback | Done (in `llmserv_complete_stream`) |
| 8.4 | Callback receives buffers with clearly-documented lifetimes | Documentation + runtime (buffer dropped after callback returns) | header comment on the typedef | Done |
| 8.5 | Re-entrancy: caller may (or may not) call back into the library from the callback | Documentation | header comment on the typedef | Undocumented (default: UB) |

### 9. ABI stability

| # | Property | Enforcement | Verification | llmserv-ffi status |
|---|---|---|---|---|
| 9.1 | `#[repr(C)]` enums and structs frozen at 1.0 (pre-1.0: anything goes, document it) | Documentation | semver + README | pre-1.0 |
| 9.2 | Soname bump on breaking ABI change | Build system | — | Not implemented (no soname versioning yet) |
| 9.3 | Header versioning (major/minor defines) to let callers check at compile time | Documentation | `#define LLMSERV_VERSION_MAJOR ...` in header | Not implemented |

### 10. Documentation and examples

| # | Property | Enforcement | Verification | llmserv-ffi status |
|---|---|---|---|---|
| 10.1 | `README.md` in the crate describing scope, build, API surface, ownership | Documentation | review | Done |
| 10.2 | Smoke test in at least one non-Rust language exercising every public fn | Runtime (green CI) | `pytest` / GitHub Actions | Done (Python: smoke, smoke_threads, smoke_destroy, smoke_stream) |
| 10.3 | Thread-safety contract table in README | Documentation | grep | Done |
| 10.4 | "What we deliberately don't do" section (non-goals) | Documentation | README "Limitations" | Done |

---

## Using the checklist

### New FFI crate

1. Copy this doc or link it from the new crate's README.
2. For every row, note the chosen position: *structural / runtime /
   documentation / not applicable*, with rationale for the last two.
3. If a row is "not applicable" — explain why in one sentence. Most
   rows apply to most libraries; the rare N/A is worth justifying.
4. Push rows upward where possible: runtime beats documentation;
   structural beats runtime.

### Review

1. Walk the checklist against the implementation.
2. Find rows where enforcement is weaker than it could be.
3. Open backlog items for upgrades. Document the deferral reason if
   not upgrading now.

### Release readiness

1. Every "Done" row needs a verification artifact committed in the
   repo (test name, grep, CI check).
2. Every "Not implemented" row needs a backlog link or an explicit
   "shipped without this, with rationale" statement.

---

## See also

- Glossary entry "Semaphore", "RAII", "Coordinated omission" —
  `docs/3-design/glossary.md`
- llmserv-ffi README and implementation —
  `llmserv/main/features/ffi/README.md`
- Backlog items T2.1–T2.7 — `llmserv/BACKLOG.md`
