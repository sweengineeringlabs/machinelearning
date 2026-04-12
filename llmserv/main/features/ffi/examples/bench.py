#!/usr/bin/env python3
"""
FFI benchmark — measures llmserv_complete latency called directly
through the .dll/.so, for apples-to-apples comparison with the HTTP
path measured by `llmc load`.

Emits the same JSON schema as `llmc load --json` so you can diff:

    llmc load http://127.0.0.1:8080/v1/chat/completions \
      -X POST -H 'Content-Type: application/json' -d @body.json \
      -n 10 --json > http.json

    python .../bench.py --count 10 --prompt "..." --json > ffi.json

    jq '.latency_ms' http.json ffi.json

Usage:
    bench.py [--count N] [--prompt STR] [--max-tokens N]
             [--temperature F] [--json] [--rate R]

Flags mirror `llmc load` where they apply. `--rate R` enables open-loop
scheduling (CO-correct) like `llmc load --rate`. Without it, closed-loop.
"""

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_size_t, c_uint32

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
TARGET_DIR = os.path.join(REPO_ROOT, "llmserv", "target", "release")
LIB_NAME = {"win32": "llmserv.dll", "darwin": "libllmserv.dylib"}.get(sys.platform, "libllmserv.so")
LIB_PATH = os.path.join(TARGET_DIR, LIB_NAME)

if not os.path.exists(LIB_PATH):
    print(f"ERR: {LIB_PATH} not found. Build first.", file=sys.stderr)
    sys.exit(2)

lib = ctypes.CDLL(LIB_PATH)

OK, INVALID_INPUT, LOAD_FAILED, RUNTIME, PANIC, INTERNAL, DESTROYED = range(7)
ERR_NAMES = ["OK", "INVALID_INPUT", "LOAD_FAILED", "RUNTIME", "PANIC", "INTERNAL", "DESTROYED"]


class LlmHandle(ctypes.Structure):
    pass


LlmHandlePtr = POINTER(LlmHandle)

lib.llmserv_init.argtypes = [POINTER(LlmHandlePtr)]
lib.llmserv_init.restype = c_int
lib.llmserv_destroy.argtypes = [LlmHandlePtr]
lib.llmserv_destroy.restype = None
lib.llmserv_complete.argtypes = [LlmHandlePtr, c_char_p, c_uint32, c_float, POINTER(c_char_p)]
lib.llmserv_complete.restype = c_int
lib.llmserv_free_string.argtypes = [c_char_p]
lib.llmserv_free_string.restype = None


def percentile(xs: list[float], q: float) -> float:
    """Linear-interpolation percentile. q in [0, 1]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    rank = q * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def main() -> None:
    ap = argparse.ArgumentParser(description="FFI benchmark for llmserv_complete")
    ap.add_argument("-n", "--count", type=int, default=10, help="number of completions to run")
    ap.add_argument("--prompt", type=str, default="Say hello briefly.")
    ap.add_argument("-m", "--max-tokens", type=int, default=16)
    ap.add_argument("-T", "--temperature", type=float, default=0.0)
    ap.add_argument(
        "-r",
        "--rate",
        type=float,
        default=None,
        help="target RPS (open-loop, CO-correct); default closed-loop",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON matching llmc load --json")
    args = ap.parse_args()

    handle = LlmHandlePtr()
    rc = lib.llmserv_init(byref(handle))
    if rc != OK:
        print(f"init failed: {ERR_NAMES[rc]}", file=sys.stderr)
        sys.exit(1)

    prompt_bytes = args.prompt.encode("utf-8")
    latencies_ms: list[float] = []
    status_counts: dict[int, int] = {}
    error_samples: list[str] = []

    start = time.perf_counter()
    interval = (1.0 / args.rate) if args.rate else 0.0

    for i in range(args.count):
        # Open-loop scheduling: dispatch at fixed intervals; latency
        # measured from scheduled time (CO-correct).
        scheduled_at = start + interval * i if args.rate else None
        if scheduled_at is not None:
            now = time.perf_counter()
            if scheduled_at > now:
                time.sleep(scheduled_at - now)

        t_measure = scheduled_at if scheduled_at is not None else time.perf_counter()
        out = c_char_p()
        rc = lib.llmserv_complete(
            handle,
            prompt_bytes,
            c_uint32(args.max_tokens),
            c_float(args.temperature),
            byref(out),
        )
        t_end = time.perf_counter()

        if rc == OK and out.value:
            lib.llmserv_free_string(out)
            latencies_ms.append((t_end - t_measure) * 1000.0)
            # Synthesize a "status": 200 for OK, matching llmc load shape.
            status_counts[200] = status_counts.get(200, 0) + 1
        else:
            status = 503 if rc == DESTROYED else 500
            status_counts[status] = status_counts.get(status, 0) + 1
            if len(error_samples) < 5:
                error_samples.append(f"{ERR_NAMES[rc] if 0 <= rc <= 6 else rc}")

    wall = time.perf_counter() - start
    lib.llmserv_destroy(handle)

    successful = status_counts.get(200, 0)
    total = args.count
    rps = total / wall if wall > 0 else 0.0

    report = {
        "total_requests": total,
        "successful": successful,
        "errors": total - successful,
        "wall_clock_secs": wall,
        "rps": rps,
        "latency_ms": {
            "min": min(latencies_ms) if latencies_ms else 0.0,
            "p50": percentile(latencies_ms, 0.50),
            "p90": percentile(latencies_ms, 0.90),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
            "p99_9": percentile(latencies_ms, 0.999),
            "max": max(latencies_ms) if latencies_ms else 0.0,
        },
        "status_counts": {str(k): v for k, v in sorted(status_counts.items())},
        "error_samples": error_samples,
        "_transport": "ffi",  # extra field vs llmc load, aids diff
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"\n=== RESULTS (FFI direct) ===")
    print(f"Total requests:  {total}")
    print(f"Successful:      {successful}")
    print(f"Errors:          {total - successful}")
    print(f"Wall clock:      {wall:.2f}s")
    print(f"Requests/sec:    {rps:.2f}")
    print(f"\n=== LATENCY (ms) ===")
    L = report["latency_ms"]
    print(f"  min:   {L['min']:>8.2f}")
    print(f"  p50:   {L['p50']:>8.2f}")
    print(f"  p90:   {L['p90']:>8.2f}")
    print(f"  p95:   {L['p95']:>8.2f}")
    print(f"  p99:   {L['p99']:>8.2f}")
    print(f"  p99.9: {L['p99_9']:>8.2f}")
    print(f"  max:   {L['max']:>8.2f}")
    print(f"\n=== STATUS CODES ===")
    for code, count in report["status_counts"].items():
        print(f"  {code}: {count}")
    if error_samples:
        print(f"\n=== ERROR SAMPLES ===")
        for e in error_samples:
            print(f"  {e}")


if __name__ == "__main__":
    main()
