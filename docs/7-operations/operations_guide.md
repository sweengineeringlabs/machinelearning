# SwellMD Operations Guide

> **TLDR:** Day-to-day operations guide for the `swellmd` HTTP inference daemon — startup, monitoring, troubleshooting, and production configuration.

**Audience**: Operators, SREs, developers running inference in production or staging

**WHAT**: Complete operations reference for the swellmd daemon
**WHY**: Single source of truth for running, monitoring, and maintaining the inference service
**HOW**: Practical runbooks with copy-paste commands and configuration examples

---

## Table of Contents

- [Overview](#overview)
- [Starting the Daemon](#starting-the-daemon)
  - [SafeTensors Models](#safetensors-models)
  - [GGUF Models](#gguf-models)
  - [Bind Address and Port](#bind-address-and-port)
  - [Optimization Profiles](#optimization-profiles)
- [Health Checks](#health-checks)
- [API Reference](#api-reference)
  - [GET /health](#get-health)
  - [GET /v1/models](#get-v1models)
  - [POST /v1/chat/completions](#post-v1chatcompletions)
  - [Streaming (SSE)](#streaming-sse)
- [Logging and Observability](#logging-and-observability)
- [Performance Tuning](#performance-tuning)
- [Process Management](#process-management)
- [Troubleshooting](#troubleshooting)
- [See Also](#see-also)

---

## Overview

`swellmd` is an HTTP daemon that serves LLM inference via an OpenAI-compatible REST API. It loads one model at startup, holds it in memory, and serves concurrent requests over HTTP.

**Key properties:**
- Single-model server (one model per process)
- OpenAI-compatible `/v1/chat/completions` endpoint
- Supports both blocking and SSE streaming responses
- Inference runs on `spawn_blocking` threads to avoid starving the async runtime
- CPU-only (no GPU required)

---

## Starting the Daemon

### SafeTensors Models

Load a model from HuggingFace Hub (downloads on first use, cached afterward):

```bash
swellmd --safetensors openai-community/gpt2 --port 8090
```

For gated models (Gemma, Llama), set the HuggingFace token:

```bash
export HF_TOKEN=hf_your_token_here
swellmd --safetensors google/gemma-3-1b-it --port 8090
```

### GGUF Models

Load a local GGUF file:

```bash
swellmd ./models/gemma-3-1b-it-Q4_0.gguf --port 8090
```

### Bind Address and Port

```bash
# Listen on all interfaces (for container/VM deployments)
swellmd --safetensors openai-community/gpt2 --host 0.0.0.0 --port 8080

# Localhost only (default)
swellmd --safetensors openai-community/gpt2 --host 127.0.0.1 --port 8090
```

### Optimization Profiles

Control parallelization and performance trade-offs:

```bash
# Default — rayon threshold=4096 (recommended)
swellmd --safetensors openai-community/gpt2 --opt-profile optimized

# Lower parallelization thresholds (may help on many-core machines)
swellmd --safetensors openai-community/gpt2 --opt-profile aggressive

# All optimizations off (for profiling baselines)
swellmd --safetensors openai-community/gpt2 --opt-profile baseline
```

### Startup Sequence

On startup, swellmd logs each phase:

```
[INFO  swellmd::core::loader] Using cached model: openai-community/gpt2
[INFO  swellmd::core::loader]   Config: arch=gpt2, dim=768, layers=12, heads=12, vocab=50257
[INFO  swellmd::core::loader]   160 tensors loaded
[INFO  swellmd::core::loader]   Quantized 73 linear layers F32 -> Q8_0
[INFO  swellmd::core::loader]   Warmup: 701ms
[INFO  swellmd::core::loader]   Model ready: 163.0M params
[INFO  swellmd::core::loader]   Tokenizer: 50257 tokens (tokenizer.json)
[INFO  swellmd] swellmd serving model 'openai-community/gpt2' on http://127.0.0.1:8090
```

The daemon is ready to accept requests once the `serving model` line appears.

---

## Health Checks

```bash
curl http://localhost:8090/health
```

Response:
```json
{
  "status": "ok",
  "model": "openai-community/gpt2"
}
```

Use this endpoint for liveness/readiness probes in container orchestrators.

---

## API Reference

### GET /health

Returns daemon status and loaded model name.

**Response (200):**
```json
{"status": "ok", "model": "openai-community/gpt2"}
```

### GET /v1/models

Lists the loaded model (OpenAI-compatible format).

**Response (200):**
```json
{
  "object": "list",
  "data": [
    {"id": "openai-community/gpt2", "object": "model", "owned_by": "local"}
  ]
}
```

### POST /v1/chat/completions

Generate a chat completion. Supports blocking and streaming modes.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | array | *required* | Chat messages `[{"role":"user","content":"..."}]` |
| `max_tokens` | int | 256 | Maximum tokens to generate |
| `temperature` | float | 0.8 | Sampling temperature (0.0 = greedy) |
| `top_k` | int | null | Top-k sampling |
| `top_p` | float | null | Nucleus sampling threshold |
| `repetition_penalty` | float | null | Repetition penalty (1.0 = none) |
| `stream` | bool | false | Enable SSE streaming |
| `model` | string | "" | Informational (daemon serves one model) |

**Blocking response (200):**
```json
{
  "id": "chatcmpl-e59af57d67a241ef858fa7f7f084f1e9",
  "object": "chat.completion",
  "created": 1775281396,
  "model": "openai-community/gpt2",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello! How can I help you?"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12}
}
```

**Error response (400/500/503):**
```json
{
  "error": {
    "message": "Invalid request: messages array must not be empty",
    "type": "invalid_request_error"
  }
}
```

### Streaming (SSE)

When `"stream": true`, the daemon returns Server-Sent Events:

```bash
curl -N http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}'
```

Each token arrives as an SSE event:
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: [DONE]
```

The stream ends with `data: [DONE]`.

---

## Logging and Observability

### Log Levels

Control logging via the `RUST_LOG` environment variable:

```bash
# Standard operation — loader + request logs
RUST_LOG=swellmd=info swellmd --safetensors openai-community/gpt2

# Verbose — includes per-request generation stats
RUST_LOG=swellmd=debug swellmd --safetensors openai-community/gpt2

# Full trace — includes rustml internal operations (tensor ops, attention)
RUST_LOG=swellmd=info,rustml=trace swellmd --safetensors openai-community/gpt2
```

### Request Logging

Each completed inference logs token count and throughput:

```
[INFO  swellmd::core::router] Generated 32 tokens in 1.45s (22.1 tok/s)
```

### Key Metrics to Monitor

| Metric | Source | What it tells you |
|--------|--------|-------------------|
| Tokens/sec | Request log | Inference throughput per request |
| Warmup time | Startup log | Model readiness delay |
| Quantized layers | Startup log | Memory optimization applied |
| KV cache size | Startup log (via `sweai infer`) | Memory footprint |

---

## Performance Tuning

### Memory Considerations

| Model | Approx Memory | Notes |
|-------|---------------|-------|
| GPT-2 (124M) | ~200 MB | Q8_0 quantized at load |
| Gemma 3 1B | ~1.2 GB | GGUF Q4_0 recommended |
| Llama 2 7B | ~4 GB | GGUF Q4_0, needs 8+ GB RAM |

### Concurrency

- Inference runs on blocking threads via `tokio::task::spawn_blocking`
- Multiple requests can be in-flight, but each holds a model reference
- For CPU-bound inference, practical concurrency is limited by core count
- The async runtime remains responsive for health checks during inference

### Optimization Profile Selection

| Profile | Rayon Threshold | Use Case |
|---------|----------------|----------|
| `optimized` | 4096 | General production use (default) |
| `aggressive` | 1024 | Many-core machines (16+ cores) |
| `baseline` | disabled | Performance profiling only |

---

## Process Management

### Foreground (Development)

```bash
RUST_LOG=swellmd=info swellmd --safetensors openai-community/gpt2 --port 8090
```

### Background (Production)

```bash
# Start in background, log to file
RUST_LOG=swellmd=info swellmd --safetensors openai-community/gpt2 --port 8090 \
  > /var/log/swellmd.log 2>&1 &

# Check if running
curl -s http://localhost:8090/health

# Stop gracefully
kill $(pgrep swellmd)
```

### Systemd Unit (Linux)

```ini
[Unit]
Description=SwellMD LLM Inference Daemon
After=network.target

[Service]
Type=simple
Environment="RUST_LOG=swellmd=info"
ExecStart=/usr/local/bin/swellmd --safetensors openai-community/gpt2 --host 0.0.0.0 --port 8090
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Troubleshooting

### Daemon won't start

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Provide a GGUF model path or --safetensors <MODEL_ID>` | No model specified | Add a GGUF path or `--safetensors` flag |
| `Failed to download model` | No internet or bad model ID | Check network; verify model ID on huggingface.co |
| `Failed to load config.json` | Corrupted cache | Delete `~/.cache/rustml/hub/<model>` and retry |
| `Address already in use` | Port conflict | Use a different `--port` or stop the existing process |

### Request failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| 503 `Model not loaded` | Model failed to initialize | Check startup logs for loading errors |
| 400 `messages array must not be empty` | Bad request | Include at least one message |
| 400 `temperature must be >= 0.0` | Invalid parameter | Fix the request parameter |
| 500 `Generation failed` | Inference error | Check logs; may indicate OOM or model corruption |

### Performance issues

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Slow first request | Model warmup | Normal — subsequent requests will be faster |
| Low tokens/sec | Debug build | Rebuild with `--release` for 5-10x speedup |
| High memory | Large model + F32 | Use GGUF Q4_0 format or smaller model |
| Jittery latency | Rayon cold start | Use `--opt-profile optimized` (default) |

---

## See Also

- [User Manual](user_manual.md) — `sweai` CLI usage
- [Deployment Guide](../6-deployment/deployment_guide.md) — Building and deploying swellmd
- [Developer Guide](../4-development/developer_guide.md) — Crate architecture and contributing
- [ADR-001: Unified LlmModel](../3-design/adr/adr-001-unified-llmmodel-for-gpt2.md) — Model architecture decision
