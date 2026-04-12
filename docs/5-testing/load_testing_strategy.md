# Load Testing Strategy

**Audience**: Developers, QA, SRE
**Status**: Proposal — phases 2 and 3 are unimplemented
**Scope**: `swellmd` HTTP daemon and `swe-ml-embed` embedding server

## Goals

1. **Capacity**: know how many concurrent requests each deployment can serve at target latency.
2. **Regression detection**: catch p95 / tok-per-sec slowdowns at PR time, before they reach main.
3. **Graceful degradation**: prove the server stays up under stress and returns useful errors (not TCP resets or OOM crashes).
4. **Release baselines**: every tagged release ships with committed baseline numbers so historical drift is visible in git.

## What this replaces

`llmserv/main/features/daemon/scripts/load_test.sh` is a 50-line bash + curl burst driver. It proves "the server survives a synchronous burst" and nothing else — no percentiles, capped at ~20 concurrent by Cygwin fork limits, one workload, one endpoint.

That script stays as a fast smoke check. Everything below is the actual strategy.

## Tool choice: `oha`

[`oha`](https://github.com/hatoohs/oha) is a single Rust binary HTTP load generator. It is the right tool here because:

- Cross-platform single binary (works on Windows, Linux, macOS) — no package manager, no runtime deps.
- Native async tokio client, no fork limits.
- Built-in histograms: p50, p90, p95, p99, p99.9 out of the box.
- JSON output (`--no-tui --json-output`) for CI parsing.
- Accepts request body from file, so workloads are just JSON files in the repo.

Alternatives considered and rejected:
- **wrk2**: best-in-class for coordinated-omission-correct latency, but Lua scripting is another language in the toolchain and Windows support is poor.
- **vegeta**: great for rate-targeted tests but Go runtime and no native Windows binaries.
- **k6**: powerful JS scenarios but a heavier install and overkill for our workload matrix.
- **hey**: simpler than `oha` but no p99.

## Scenario taxonomy

Each scenario answers a specific question. All run against one `swellmd` instance with a known `--max-concurrent` setting.

| Scenario | Question | Shape | Duration |
|---|---|---|---|
| `smoke`    | Does the server respond at all? | 1 req | seconds |
| `baseline` | What is steady-state latency at low load? | constant low RPS | 60s |
| `ramp`     | At what RPS does p95 cross the SLO? | step up RPS until errors | 5–10 min |
| `stress`   | Does the server survive beyond capacity? | 10× capacity RPS | 60s |
| `spike`    | Does latency recover after a burst? | 0 → N → 0 step | 60s |
| `soak`     | Does memory / FDs leak over hours? | modest RPS | 1–8 h |

The `stress` scenario is specifically the regression guard for the OOM bug that motivated this work — with admission control, we expect HTTP 503s, not process death.

## Workload matrix

Four request shapes cover realistic usage. Stored as JSON bodies in `scripts/loadtest/workloads/`.

| Workload | Prompt tokens | max_tokens | Endpoint | Streaming |
|---|---|---|---|---|
| `chat_short`     | ~10   | 32  | `/v1/chat/completions` | no  |
| `chat_medium`    | ~200  | 128 | `/v1/chat/completions` | no  |
| `chat_long`      | ~1000 | 256 | `/v1/chat/completions` | no  |
| `chat_streaming` | ~50   | 128 | `/v1/chat/completions` | yes |
| `embeddings_small` | ~20, batch 1 | n/a | `/v1/embeddings` | no |
| `embeddings_batch` | ~50, batch 16 | n/a | `/v1/embeddings` | no |

Scenario × workload = 36 cells. We don't run the full matrix every commit — see phase plan.

## Metrics

Captured per scenario, written to `scripts/loadtest/results/<ts>/<scenario>.<workload>.json`.

**HTTP** (from oha):
- Total requests, successful requests, error breakdown by status code
- RPS achieved
- Latency histogram: p50, p90, p95, p99, p99.9, max

**Inference-specific** (parsed from response JSON):
- Tokens per second per request (`completion_tokens / duration`)
- Total tokens per second across all requests

**Streaming only** (computed client-side by a small companion tool, since oha doesn't parse SSE):
- Time to first token (TTFT)
- Inter-token latency

**System** (sampled by a sidecar script at 1 Hz):
- CPU utilization
- Resident set size (RSS)
- Open file descriptors
- Loaded via `psutil` or platform equivalents

## Directory layout

```
llmserv/main/features/daemon/scripts/
├── load_test.sh                    # keep: fast smoke (existing)
└── loadtest/
    ├── run.sh                      # entrypoint: run.sh <scenario> <workload>
    ├── scenarios/
    │   ├── smoke.env               # RPS=1, duration=5s
    │   ├── baseline.env
    │   ├── ramp.env
    │   ├── stress.env
    │   ├── spike.env
    │   └── soak.env
    ├── workloads/
    │   ├── chat_short.json
    │   ├── chat_medium.json
    │   ├── chat_long.json
    │   ├── chat_streaming.json
    │   ├── embeddings_small.json
    │   └── embeddings_batch.json
    ├── monitor.sh                  # system metrics sidecar
    ├── compare.sh                  # diff results vs baseline, exit nonzero on regression
    └── baselines/
        ├── v0.8.0.json             # committed per tagged release
        └── ...
```

One runner script, declarative scenarios and workloads, reproducible outputs.

## Baseline and regression detection

Each tagged release regenerates baselines on one designated host:

1. Check out the tag, build `--release`.
2. Run `run.sh baseline chat_medium` and `run.sh baseline chat_streaming`.
3. Write `baselines/<tag>.json` with p50/p95/p99 and tok/s.
4. Commit to the repo.

PR checks then run a short baseline scenario (60s) and call `compare.sh <latest_baseline>`. The check fails if:
- p95 regressed more than 20%
- throughput dropped more than 10%
- error rate > 0 outside expected 503s

Thresholds are a starting point — tune after the first month of data.

## CI integration

Phase 3 goal. Approximate shape:

```yaml
load-smoke:
  needs: [build-release]
  steps:
    - run: ./llmserv/main/features/daemon/scripts/loadtest/run.sh baseline chat_short
    - run: ./llmserv/main/features/daemon/scripts/loadtest/compare.sh latest
```

Runs only when `llmserv/main/features/daemon/`, `llmserv/main/features/inference/layers/`, `llmserv/main/features/inference/model/`, or `main/features/tensor/` change, since those are the hot paths. Full matrix runs nightly on a dedicated runner, results published as a build artifact.

## Non-goals

Stated explicitly so future contributors don't re-ask:

- **Distributed load generation**: one client host for now. If per-client CPU becomes the bottleneck we will revisit; the 1B model on CPU runs nowhere near line rate.
- **Realistic traffic replay**: no Poisson arrivals, no recorded production traces. Synthetic workloads only.
- **Chaos testing**: no network partition, CPU throttling, disk fill tests. Those belong to an SRE track.
- **Cross-model parity**: each model has its own baseline; we do not compare absolute numbers across model sizes.

## Phased delivery

### Phase 1 — foundation (1–2 days of work)

- Install `oha` (document in prereqs).
- Create `scripts/loadtest/` skeleton.
- Write `baseline` and `stress` scenarios only.
- Write `chat_short` and `chat_medium` workloads.
- Run both scenarios against v0.8.0, commit baseline numbers.
- Update `docs/5-testing/report/load_testing_2026_04_12_admission_control.md` (or add a new dated report) with the v0.8.0 baseline.

Deliverable: reproducible numbers checked into the repo.

### Phase 2 — coverage (1 week)

- Add `ramp`, `spike`, `soak` scenarios.
- Add remaining workloads (long, streaming, embeddings).
- Write `monitor.sh` and `compare.sh`.
- Document regression thresholds.

Deliverable: full scenario × workload matrix runnable locally.

### Phase 3 — CI integration (1–2 weeks)

- GitHub Actions workflow for load-smoke on PRs to hot paths.
- Nightly full-matrix run with artifact publishing.
- Slack / email on regression.
- Automatic baseline PR on tagged release.

Deliverable: load regressions blocked at PR time.

## Open questions

- **Runner hardware**: CI GitHub runners are noisy. Numbers are only comparable on the same machine — we need a dedicated host or at minimum a single known runner class for baselines.
- **Streaming parser**: oha does not parse SSE. Do we write a small Rust client for TTFT, or accept that streaming latency stays a manual/report-only metric?
- **Model fixture**: `google/gemma-3-1b-it` is the reference model today. If it changes, historical baselines become incomparable — tag the baseline with the model ID + commit.
- **Warmup**: each scenario should discard the first 5–10 seconds to exclude JIT/cache warmup effects. Oha does not support this natively; either two-phase the scenario or post-process the histogram.

## Success criteria

The strategy is working when:

1. Every tagged release has committed baseline numbers for at least `baseline.chat_short` and `baseline.chat_medium`.
2. At least one PR this quarter gets blocked for a p95 regression before merge.
3. A burst of 100 concurrent requests against the daemon does not crash the process — only returns 503s.
4. On-call can answer "what is our p99 for medium prompts at 5 concurrent?" from a committed artifact, not by re-running a test.
