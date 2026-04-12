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
| `llmserv_complete(h, prompt, max_tokens, temperature, out_text**)` | Single completion (blocking). |
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

The contract is **strong**. It's defensible because every type under the
handle (`LlmModel`, `Generator`, all tokenizers) has no interior
mutability — model weights, configs, and vocabularies are read-only, and
each generation allocates its own KV cache + activations. The daemon
exercises the same pattern (one shared model behind `Arc`, many
concurrent requests) in production.

**What is safe:**

| Operation | From one thread | From N threads on the same handle | On multiple handles across N threads |
|---|---|---|---|
| `llmserv_complete` | safe | **safe** (concurrent calls OK) | safe |
| `llmserv_embed` | safe | **safe** | safe |
| `llmserv_tokenize` | safe | **safe** | safe |
| `llmserv_token_count` | safe | **safe** | safe |
| `llmserv_free_*` | safe | safe (different buffers) | safe |

**What is NOT safe:**

- **`llmserv_destroy` concurrent with any other call on the same handle.**
  Destroy invalidates the memory. Caller must ensure no other thread is
  inside `llmserv_*(h, ...)` when `llmserv_destroy(h)` is called. Typical
  pattern: join worker threads before destroy, or wrap the handle in
  the caller's own lifetime-managing type.
- **Double-destroy.** Calling `llmserv_destroy` twice on the same handle
  is undefined behavior. Treat the handle pointer as invalid after the
  first destroy.
- **Calling any function on a handle after destroy.** Use-after-free.
- **Freeing a Rust-allocated buffer with anything other than the
  matching `llmserv_free_*`.** In particular, do not pass strings to
  `free()` or `PyMem_Free` from Python.

**What is *allowed but not useful*:**

- Spawning N threads that each call `llmserv_complete` on the same
  handle does NOT give N× throughput. The underlying matmul uses rayon's
  global thread pool, so all N calls contend for the same CPUs. See the
  load-testing report for why concurrency ≈ 2 is the sweet spot on CPU.

**Rule of thumb:** share one handle across threads for read-only
concurrent use; create the handle before spawning workers; destroy
only after every worker has joined.

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
  (Python asyncio threadpool, Go goroutine, etc.).
- **No streaming.** `llmserv_complete` returns the full text; there's no
  token-by-token callback yet. Planned as T2.5.
- **Single handle per model load.** Multiple handles in the same process
  load weights independently — no `Arc<Mmap>` sharing. Not a bottleneck
  for desktop use.
- **Not ABI-stable across versions.** The header is regenerated per
  build and breaking changes are allowed pre-1.0. Pin to a tag.
