# Embedding Throughput Baseline

> **TLDR:** Reference embedding throughput on known hardware. All future optimizations are measured against these numbers.

**Status**: NOT YET RUN — this document is a template. Fill in after running the benchmark suite below.

---

## Hardware

| Component | Value |
|-----------|-------|
| CPU | _(fill in)_ |
| Threads | _(fill in)_ |
| RAM | _(fill in)_ |
| OS | _(fill in)_ |
| Platform | _(fill in)_ |

---

## Benchmark procedure

```bash
# Build release binary
cargo build --release

# Start daemon with a real model
APPDATA=./bench-config cargo run --release --bin embed -- serve

# In another terminal: drive with grpcurl (install once: go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest)
# Single embed
grpcurl -plaintext -d '{"inputs":["The quick brown fox"]}' \
  127.0.0.1:8181 justembed.EmbedService/Embed

# Batch embed (8 inputs)
grpcurl -plaintext -d '{"inputs":["text 1","text 2","text 3","text 4","text 5","text 6","text 7","text 8"]}' \
  127.0.0.1:8181 justembed.EmbedService/Embed

# Throughput: N sequential single-input calls, measure wall time
time for i in $(seq 1 100); do
  grpcurl -plaintext -d '{"inputs":["The quick brown fox"]}' \
    127.0.0.1:8181 justembed.EmbedService/Embed > /dev/null
done
```

---

## Results

### Single-input latency

| Model | Format | Quant | Input tokens | Build | P50 ms | P99 ms |
|-------|--------|-------|-------------|-------|--------|--------|
| nomic-bert | GGUF | _(fill in)_ | ~8 | release | — | — |
| nomic-bert | GGUF | _(fill in)_ | ~64 | release | — | — |
| nomic-bert | GGUF | _(fill in)_ | ~256 | release | — | — |

### Batch throughput (embeddings/sec)

| Model | Format | Batch size | Build | emb/s |
|-------|--------|-----------|-------|-------|
| nomic-bert | GGUF | 1 | release | — |
| nomic-bert | GGUF | 8 | release | — |
| nomic-bert | GGUF | 32 | release | — |

---

## Notes

_(Fill in after running: SIMD dispatch confirmation, thread count, any anomalies.)_

- [ ] Confirm SIMD dispatch: check startup log for AVX2 / NEON activation
- [ ] Confirm rayon thread count matches hardware thread count
- [ ] Verify L2-norm of output vectors ≈ 1.0 (spot-check first result)
