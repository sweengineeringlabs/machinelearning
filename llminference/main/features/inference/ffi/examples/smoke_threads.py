#!/usr/bin/env python3
"""
Thread-safety smoke test for libllmserv.

Fires N concurrent token_count calls on ONE handle from ONE Python
process using a ThreadPoolExecutor. token_count is read-only and
sub-millisecond, so this exercises the race-risk surface without
waiting for slow generations.

If the thread-safety contract holds, every call returns LlmError::Ok
and the counts are consistent across threads.

Expected output: N successes, consistent token counts per input,
clean destroy.
"""

import ctypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import POINTER, byref, c_char_p, c_int, c_size_t, c_uint32

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
TARGET_DIR = os.path.join(REPO_ROOT, "llmserv", "target", "release")
LIB_NAME = {"win32": "llmserv.dll", "darwin": "libllmserv.dylib"}.get(sys.platform, "libllmserv.so")
LIB_PATH = os.path.join(TARGET_DIR, LIB_NAME)

if not os.path.exists(LIB_PATH):
    print(f"ERR: {LIB_PATH} not found. Build first.", file=sys.stderr)
    sys.exit(2)

lib = ctypes.CDLL(LIB_PATH)


class LlmHandle(ctypes.Structure):
    pass


LlmHandlePtr = POINTER(LlmHandle)

lib.llmserv_init.argtypes = [POINTER(LlmHandlePtr)]
lib.llmserv_init.restype = c_int
lib.llmserv_destroy.argtypes = [LlmHandlePtr]
lib.llmserv_destroy.restype = None
lib.llmserv_token_count.argtypes = [LlmHandlePtr, c_char_p, POINTER(c_size_t)]
lib.llmserv_token_count.restype = c_int


def token_count(handle: LlmHandlePtr, text: bytes) -> int:
    out = c_size_t()
    rc = lib.llmserv_token_count(handle, text, byref(out))
    if rc != 0:
        raise RuntimeError(f"token_count rc={rc}")
    return out.value


def main() -> None:
    print(f"Loading library: {LIB_PATH}")
    handle = LlmHandlePtr()
    rc = lib.llmserv_init(byref(handle))
    if rc != 0:
        print(f"init failed: rc={rc}", file=sys.stderr)
        sys.exit(1)
    print("  init OK")

    # Baseline: single-threaded counts per input.
    inputs = [b"hello", b"hello world", b"Hello, world!", b"the quick brown fox"]
    baseline = {t: token_count(handle, t) for t in inputs}
    print(f"  baseline counts: {baseline}")

    # Concurrency: N threads × M calls each, all against the same handle.
    N_THREADS = 16
    CALLS_PER_THREAD = 250
    total = N_THREADS * CALLS_PER_THREAD
    print(f"  firing {total} calls across {N_THREADS} threads...")

    def worker(tid: int) -> tuple[int, int]:
        successes = 0
        mismatches = 0
        for i in range(CALLS_PER_THREAD):
            t = inputs[(tid + i) % len(inputs)]
            c = token_count(handle, t)
            if c == baseline[t]:
                successes += 1
            else:
                mismatches += 1
        return successes, mismatches

    with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
        results = [f.result() for f in as_completed([ex.submit(worker, t) for t in range(N_THREADS)])]

    total_ok = sum(s for s, _ in results)
    total_err = sum(m for _, m in results)
    print(f"  successes: {total_ok}, mismatches: {total_err}")
    assert total_ok == total, f"expected {total} successful calls, got {total_ok}"
    assert total_err == 0, "token counts disagreed between threads — handle is not thread-safe!"

    lib.llmserv_destroy(handle)
    print("  destroy OK")
    print(f"\nThread safety confirmed: {total}/{total} concurrent calls produced consistent results.")


if __name__ == "__main__":
    main()
