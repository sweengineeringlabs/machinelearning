# SwellMD Deployment Guide

> **TLDR:** Build, package, and deploy the `swellmd` inference daemon — from local development to containerized production.

**Audience**: Developers, DevOps, platform engineers

**WHAT**: End-to-end deployment guide for the swellmd HTTP inference daemon
**WHY**: Reproducible builds and deployments for local, staging, and production environments
**HOW**: Step-by-step instructions covering build, configuration, container, and verification

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building from Source](#building-from-source)
  - [Debug Build](#debug-build)
  - [Release Build](#release-build)
  - [Build Verification](#build-verification)
- [Binary Deployment](#binary-deployment)
  - [Standalone Binary](#standalone-binary)
  - [Required Runtime Files](#required-runtime-files)
- [Configuration](#configuration)
  - [CLI Arguments](#cli-arguments)
  - [Environment Variables](#environment-variables)
- [Model Provisioning](#model-provisioning)
  - [SafeTensors (HuggingFace Hub)](#safetensors-huggingface-hub)
  - [GGUF (Local Files)](#gguf-local-files)
  - [Pre-caching Models](#pre-caching-models)
- [Container Deployment](#container-deployment)
  - [Dockerfile](#dockerfile)
  - [Docker Compose](#docker-compose)
- [Deployment Verification](#deployment-verification)
  - [Smoke Tests](#smoke-tests)
  - [Load Test](#load-test)
- [Architecture Compatibility](#architecture-compatibility)
- [Supported Models](#supported-models)
- [See Also](#see-also)

---

## Prerequisites

- **Rust toolchain**: 1.85+ (edition 2024)
- **Platform**: Linux (x86_64, aarch64), Windows (x86_64), macOS (x86_64, aarch64)
- **RAM**: 512 MB minimum (GPT-2), 2+ GB recommended (1B+ models)
- **CPU**: AVX2 support recommended for SIMD-accelerated inference

Check AVX2 support:
```bash
# Linux
grep avx2 /proc/cpuinfo

# Windows (PowerShell)
(Get-CimInstance Win32_Processor).Caption
```

---

## Building from Source

### Debug Build

Fast compilation, slower inference (5-10x slower than release):

```bash
cd /path/to/machinelearning
cargo build -p swellmd
# Binary: ./target/debug/swellmd
```

### Release Build

Optimized for production inference (thin LTO enabled):

```bash
cargo build -p swellmd --release
# Binary: ./target/release/swellmd
```

Typical build times:
- Debug: ~3-4 minutes (first build), ~30s (incremental)
- Release: ~8-12 minutes (first build), ~2 minutes (incremental)

### Build Verification

```bash
# Check binary exists and runs
./target/release/swellmd --help

# Expected output:
# swellmd — HTTP daemon for RustML LLM inference.
# ...
```

---

## Binary Deployment

### Standalone Binary

The `swellmd` binary is fully self-contained. Deploy it by copying the single binary:

```bash
# Linux
cp target/release/swellmd /usr/local/bin/

# Windows
copy target\release\swellmd.exe C:\tools\
```

### Required Runtime Files

No additional runtime files are needed. Model files are either:
- Downloaded from HuggingFace on first use (SafeTensors path)
- Provided as local files (GGUF path)

**Cache directory** (auto-created): `~/.cache/llm/hub/`

---

## Configuration

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `<GGUF_PATH>` | — | Path to a GGUF model file (positional) |
| `--safetensors <ID>` | — | HuggingFace model ID (e.g. `openai-community/gpt2`) |
| `--host <ADDR>` | `127.0.0.1` | Bind address |
| `--port <PORT>` | `8080` | Listen port |
| `--opt-profile <P>` | `optimized` | Performance profile: `optimized`, `aggressive`, `baseline` |

One of `GGUF_PATH` or `--safetensors` is required.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Log level filter (e.g. `swellmd=info`, `swellmd=debug,rustml=trace`) |
| `HF_TOKEN` | HuggingFace API token for gated models (Gemma, Llama) |
| `RAYON_NUM_THREADS` | Override number of rayon worker threads |

---

## Model Provisioning

### SafeTensors (HuggingFace Hub)

Point `application.toml` at a HuggingFace repo ID:

```toml
[model]
source = "safetensors"
id = "openai-community/gpt2"
```

Then:

```bash
# First run — downloads model (~500 MB for GPT-2)
swellmd

# Subsequent runs — uses cache
swellmd
```

**Cache location**: `~/.cache/llm/hub/<org>--<model>/`

**Gated models** require authentication via env:

```bash
export HF_TOKEN=hf_your_token_here
swellmd
```

### GGUF (Local Files)

```toml
[model]
source = "gguf"
path = "/models/gemma-3-1b-it-Q4_0.gguf"
```

### Pre-caching Models

For air-gapped or container deployments, pre-download models:

```bash
# Use the sweai CLI to download
sweai hub download openai-community/gpt2

# Verify cache
sweai hub list
# openai-community/gpt2    C:\Users\you\.cache\rustml\hub\openai-community--gpt2

# Point application.toml [model].id at it and start daemon (no network)
swellmd
```

---

## Container Deployment

### Dockerfile

```dockerfile
# Stage 1: Build
FROM rust:1.85-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build -p swellmd --release

# Stage 2: Runtime
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/swellmd /usr/local/bin/

# application.toml is baked into the swellmd binary as the bundled
# default. To override, mount an XDG config at /etc/config/llmserv/application.toml
# and set XDG_CONFIG_HOME=/etc/config in the container.

EXPOSE 8080
ENV XDG_CONFIG_HOME=/etc/config

ENTRYPOINT ["swellmd"]
```

An override TOML mounted at `/etc/config/llmserv/application.toml` might look like:

```toml
[server]
host = "0.0.0.0"
port = 8080

[model]
source = "safetensors"
id = "openai-community/gpt2"
```

**Build and run:**

```bash
docker build -t swellmd .

# With config and models mounted
docker run -p 8080:8080 \
  -v /path/to/config:/etc/config \
  -v /path/to/models:/models \
  swellmd

# With SafeTensors auto-download + persistent cache
docker run -p 8080:8080 \
  -v /path/to/config:/etc/config \
  -v swellmd-cache:/root/.cache/rustml \
  -e HF_TOKEN=hf_xxx \
  swellmd
```

### Docker Compose

```yaml
services:
  swellmd:
    build: .
    ports:
      - "8090:8080"
    volumes:
      - ./config:/etc/config                 # contains llmserv/application.toml
      - ./models:/models
      - swellmd-cache:/root/.cache/rustml
    environment:
      - XDG_CONFIG_HOME=/etc/config
      - HF_TOKEN=${HF_TOKEN}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
    deploy:
      resources:
        limits:
          memory: 2G

volumes:
  swellmd-cache:
```

---

## Deployment Verification

### Smoke Tests

Run these after every deployment to verify the daemon is operational:

```bash
BASE_URL="http://localhost:8090"

# 1. Health check
curl -sf "$BASE_URL/health" | grep '"ok"'
echo "PASS: health"

# 2. Model listing
curl -sf "$BASE_URL/v1/models" | grep '"model"'
echo "PASS: models"

# 3. Blocking completion
curl -sf "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":8}' \
  | grep '"assistant"'
echo "PASS: completion"

# 4. Streaming completion
curl -sfN "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"stream":true}' \
  | grep -m1 'chat.completion.chunk'
echo "PASS: streaming"

# 5. Validation (bad request)
STATUS=$(curl -so /dev/null -w '%{http_code}' "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[],"max_tokens":8}')
[ "$STATUS" = "400" ] && echo "PASS: validation" || echo "FAIL: expected 400, got $STATUS"
```

### Load Test

Basic concurrent load test with `curl` in parallel:

```bash
# 10 concurrent requests
for i in $(seq 1 10); do
  curl -s http://localhost:8090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Count to 5"}],"max_tokens":16}' &
done
wait
echo "All requests completed"
```

---

## Architecture Compatibility

| Platform | SIMD | Status |
|----------|------|--------|
| Linux x86_64 (AVX2) | AVX2 | Primary target |
| Linux x86_64 (SSE4.1) | SSE4.1 | Supported, slower matmul |
| Linux aarch64 | NEON | Supported |
| Windows x86_64 | AVX2 | Tested |
| macOS x86_64 | AVX2 | Supported |
| macOS aarch64 (Apple Silicon) | NEON | Supported |

---

## Supported Models

Models tested with swellmd:

| Model | Format | Size | Notes |
|-------|--------|------|-------|
| `openai-community/gpt2` | SafeTensors | 124M | Learned positional embedding, LayerNorm |
| `google/gemma-3-1b-it` | SafeTensors / GGUF | 1B | RoPE, RMSNorm, GeGLU (requires HF token) |
| Llama 2 / Mistral | GGUF | 7B+ | RoPE, SwiGLU, GQA |
| Falcon | GGUF | 7B+ | ALiBi or RoPE variant |
| Mixtral | GGUF | 8x7B | Mixture of Experts |
| Nomic-BERT | GGUF | 137M | Encoder-only |

**OpenAI-compatible clients** (Python `openai`, JS, etc.) work by pointing `base_url` to `http://host:port/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8090/v1", api_key="unused")
response = client.chat.completions.create(
    model="openai-community/gpt2",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

---

## See Also

- [Operations Guide](../7-operations/operations_guide.md) — Day-to-day daemon operations
- [Developer Guide](../4-development/developer_guide.md) — Crate architecture and contributing
- [User Manual](../7-operations/user_manual.md) — `sweai` CLI usage
- [Model Verification Guide](../4-development/guides/model-verification.md) — Verifying model correctness
