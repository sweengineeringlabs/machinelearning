# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
cargo build --release

# Run all tests (unit + integration)
cargo test --workspace

# Run a single test by name
cargo test --workspace test_<name>

# Run only unit tests for one crate
cargo test -p swe-ml-embedding

# Run only integration tests
cargo test -p swe-embedding-systemd --test grpc_embed_int_test
cargo test -p swe-embedding-systemd --test grpc_health_int_test

# Run the daemon (requires gguf_path set in config)
cargo run --bin embed -- serve
```

There is no Makefile or linting script. Proto types are generated at compile time via `systemd/build.rs` using `tonic-build` — changes to `systemd/proto/embed.proto` take effect on the next `cargo build`.

## Configuration

The daemon uses XDG config layering (later overrides earlier):

1. Bundled default (`llminference/main/config/application.toml`)
2. System-wide: `$XDG_CONFIG_DIRS/llminference/application.toml`
3. User override: `$XDG_CONFIG_HOME/llminference/application.toml` (Windows: `%APPDATA%\llminference\application.toml`)

Key config sections (see `systemd/src/api/config.rs`):

```toml
[embedding.model]
gguf_path = "/path/to/model.gguf"   # empty string = no-model mode

[embedding.grpc]
host = "127.0.0.1"
port = 8181
max_message_bytes = 4194304         # 4 MiB

# Optional mTLS
[embedding.grpc.tls]
cert_pem_path = "..."
key_pem_path = "..."
client_ca_pem_path = "..."          # omit for server-TLS-only
```

## External crate dependencies (path deps)

These are workspace siblings, not published crates:

| Alias | Local path |
|-------|-----------|
| `swe-llmmodel-*` | `../llmserve/llmmodel/` |
| `swe-edge-ingress-grpc` | `../edge/ingress/grpc` |
| `swe-systemd` | `../llminference/main/features/systemd` |

If a build fails with "package not found", check that those sibling directories are present at the expected relative paths.

## Testing conventions

- Test names follow `test_<action>_<condition>_<expectation>`.
- Unit tests live in the same file as the code under test.
- Integration tests in `systemd/tests/` use a real in-process gRPC server bound to a random port — no mocks.
- Content-correctness is required: latency or "didn't crash" alone is not sufficient to claim a backend works.

## Security

See `docs/3-design/threat_model.md` for the full analysis. The defaults are dev-friendly (localhost, plaintext, no auth). For production: set TLS via config, enforce network policy for auth (T6 is not wired), and use a read-only GGUF mount with SHA256 pinning.
