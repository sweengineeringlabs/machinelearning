# llmserv-ffi

C-ABI cdylib for llmserv — load the inference stack directly from
Python / Go / C# / Swift / Java / any language with FFI support.

**Not a performance play.** LLM generation is compute-bound (~3000 ms on
CPU for a 1 B model). HTTP loopback overhead is ~1 ms — 0.03% of wall
time. FFI does not make inference faster. See T2 in
`llmserv/BACKLOG.md` for the honest motivation.

**What FFI buys you:** no port binding, no process lifecycle management,
single-binary distribution, works in sandboxed extension hosts (IDE
plugins). These are the real reasons to use it.

## Build

```bash
cargo build --release --manifest-path llmserv/Cargo.toml -p llmserv-ffi
```

Artifacts land at `llmserv/target/release/`:

| Platform | File |
|---|---|
| Linux | `libllmserv.so` |
| macOS | `libllmserv.dylib` |
| Windows | `llmserv.dll` |

The C header is regenerated on every build into `include/llmserv.h`.
Commit it so non-Rust consumers don't need a Rust toolchain to get
bindings.

## API surface

See `include/llmserv.h` for the canonical declarations. Summary:

| Function | Purpose |
|---|---|
| `llmserv_init(LlmHandle**)` | Load the model specified by `application.toml`. |
| `llmserv_complete(h, prompt, max_tokens, temperature, out_text**)` | Single completion (blocking, returns full text). |
| `llmserv_complete_stream(h, prompt, max_tokens, temperature, cb, ctx)` | Streaming completion — `cb` called per token; return `false` to stop. |
| `llmserv_embed(h, text, out_vec**, out_dim*)` | Mean-pooled embedding. |
| `llmserv_tokenize(h, text, out_ids**, out_len*)` | Encode to token ids. |
| `llmserv_token_count(h, text, out_count*)` | Just the count (IDE keystroke-frequency use). |
| `llmserv_destroy(h)` | Free the handle. |
| `llmserv_free_string(char*)` | Free a string returned by `complete`. |
| `llmserv_free_floats(float*, usize)` | Free floats returned by `embed`. |
| `llmserv_free_u32s(u32*, usize)` | Free ids returned by `tokenize`. |

All functions return an `LlmError` code (zero = OK). Panics are caught
at the boundary and converted to `PANIC`; they never unwind across FFI.

## Memory ownership

**Rust allocates, Rust frees.** Every pointer written to an `out_*`
argument is valid until you call the matching `llmserv_free_*` function.
Do not call your platform's `free()` on these pointers. Do not use them
after `llmserv_destroy`.

## Thread-safety contract

The contract is **strong** — the implementation eliminates every
traditional C-ABI race by construction. Every type under the handle
(`LlmModel`, `Generator`, all tokenizers) has no interior mutability;
weights and vocabularies are read-only; each generation allocates its
own KV cache and activations. The handle itself is wrapped in an
`RwLock<Option<Model>>`: calls take a read-lock (many concurrent),
destroy takes a write-lock (waits for all readers).

### What is safe

| Operation | From one thread | From N threads on the same handle | On multiple handles across N threads |
|---|---|---|---|
| `llmserv_complete` | safe | **safe** (concurrent calls OK) | safe |
| `llmserv_embed` | safe | **safe** | safe |
| `llmserv_tokenize` | safe | **safe** | safe |
| `llmserv_token_count` | safe | **safe** | safe |
| `llmserv_free_*` | safe | safe (different buffers) | safe |

### Destroy, use-after-destroy, double-destroy — all safe

The handle pointer returned by `llmserv_init` remains valid for the
lifetime of the process. `llmserv_destroy` releases the inner model
(frees weights, activations, tokenizer) but does not free the outer
handle struct. This means:

- **`llmserv_destroy` concurrent with any other call** — the destroy
  waits for in-flight calls to finish, then transitions atomically.
  In-flight calls complete normally; calls that start after destroy
  return `LlmError::Destroyed`.
- **Double-destroy** — a safe no-op. Second and subsequent destroys
  find the inner already gone and return without error.
- **Use-after-destroy** — returns `LlmError::Destroyed` instead of
  undefined behavior. No segfault, no memory corruption.

Per-init memory cost: one small struct (~40 bytes) permanently retained
after destroy. Bounded by the number of `llmserv_init` calls in the
process lifetime — irrelevant for typical desktop/IDE usage where a
handle lives for the process.

### Still the caller's responsibility

- **Free Rust-allocated buffers with the matching `llmserv_free_*`**,
  not `free()` / `PyMem_Free` / `delete`. The allocator crossing the
  FFI boundary must match. Opaque wrapper classes in the host language
  are the idiomatic fix (see the Python smoke tests for the pattern).
- **Respect the ownership lifetime of returned buffers.** A pointer
  from `llmserv_complete` is valid until you call `llmserv_free_string`
  on it — not after.

### Allowed but not useful

Spawning N threads that each call `llmserv_complete` on the same handle
does NOT give N× throughput. The underlying matmul uses rayon's global
thread pool, so all calls contend for the same CPUs. See the load-test
report for why concurrency ≈ 2 is the sweet spot on CPU.

### Rule of thumb

Share one handle freely across threads. Destroy whenever — the library
handles synchronization. Free returned buffers with the typed free
functions.

## Python example

See `examples/smoke.py` for the canonical `ctypes` binding pattern. Run
it:

```bash
python llmserv/main/features/ffi/examples/smoke.py
```

Expected output: init → token_count → tokenize → complete → destroy,
all returning `OK`.

## Configuration

`llmserv_init` reads the same `application.toml` that the daemons use,
with the same XDG load order:

1. Bundled default (compiled into the library)
2. `$XDG_CONFIG_DIRS/llmserv/application.toml`
3. `$XDG_CONFIG_HOME/llmserv/application.toml`

To point the FFI at a specific model without editing the bundled
default, set `XDG_CONFIG_HOME` to a directory containing an override:

```
$XDG_CONFIG_HOME/llmserv/application.toml:
    [model]
    source = "gguf"
    path = "/path/to/model.gguf"
```

## Limitations (today)

- **Blocking only.** No async FFI; callers wrap in their own async
  (Python asyncio threadpool, Go goroutine, etc.). `llmserv_complete_stream`
  is still blocking from the caller's thread — it just reports progress
  via the callback as it goes.
- **Single handle per model load.** Multiple handles in the same process
  load weights independently — no `Arc<Mmap>` sharing. Not a bottleneck
  for desktop use.
- **Not ABI-stable across versions.** The header is regenerated per
  build and breaking changes are allowed pre-1.0. Pin to a tag.
