#!/usr/bin/env python3
"""
Streaming-completion smoke test.

Verifies that llminference_complete_stream invokes the callback per token,
piece-by-piece, instead of blocking until the full text is ready.
This is what an IDE autocomplete UI needs: show each token as it lands.

Also exercises:
    - returning False from the callback stops generation early
    - user_data round-trips correctly (opaque pointer)
"""

import ctypes
import os
import sys
import time
from ctypes import (
    CFUNCTYPE,
    POINTER,
    byref,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_uint32,
    c_void_p,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
TARGET_DIR = os.path.join(REPO_ROOT, "llminference", "target", "release")
LIB_NAME = {"win32": "llminference.dll", "darwin": "libllminference.dylib"}.get(sys.platform, "libllminference.so")
LIB_PATH = os.path.join(TARGET_DIR, LIB_NAME)

if not os.path.exists(LIB_PATH):
    print(f"ERR: {LIB_PATH} not found.", file=sys.stderr)
    sys.exit(2)

lib = ctypes.CDLL(LIB_PATH)

OK, INVALID_INPUT, LOAD_FAILED, RUNTIME, PANIC, INTERNAL, DESTROYED = range(7)


class LlmHandle(ctypes.Structure):
    pass


LlmHandlePtr = POINTER(LlmHandle)

lib.llminference_init.argtypes = [POINTER(LlmHandlePtr)]
lib.llminference_init.restype = c_int
lib.llminference_destroy.argtypes = [LlmHandlePtr]
lib.llminference_destroy.restype = None

# The callback signature: bool(*)(const char*, void*)
TokenCallback = CFUNCTYPE(c_bool, c_char_p, c_void_p)

lib.llminference_complete_stream.argtypes = [
    LlmHandlePtr,
    c_char_p,
    c_uint32,
    c_float,
    TokenCallback,
    c_void_p,
]
lib.llminference_complete_stream.restype = c_int


def main() -> None:
    print(f"Loading library: {LIB_PATH}")
    handle = LlmHandlePtr()
    rc = lib.llminference_init(byref(handle))
    assert rc == OK, f"init rc={rc}"
    print("  init OK")

    # Scenario 1: receive full completion token by token.
    print("  scenario 1: streaming receives tokens progressively")
    pieces: list[str] = []
    arrival_times: list[float] = []
    t0 = time.time()

    @TokenCallback
    def on_token(piece, _user_data):
        arrival_times.append(time.time() - t0)
        pieces.append(piece.decode("utf-8"))
        return True  # keep going

    rc = lib.llminference_complete_stream(
        handle,
        b"Count to five:",
        c_uint32(20),
        c_float(0.0),
        on_token,
        None,
    )
    assert rc == OK, f"stream rc={rc}"
    assembled = "".join(pieces)
    print(f"    received {len(pieces)} tokens in {arrival_times[-1]:.2f}s")
    print(f"    first token at {arrival_times[0]:.2f}s, last at {arrival_times[-1]:.2f}s")
    print(f"    assembled: {assembled!r}")
    assert len(pieces) > 0, "expected at least one token"
    assert arrival_times[0] < arrival_times[-1], "tokens arrived together, not streaming"

    # Scenario 2: returning False stops generation.
    print("  scenario 2: callback returns False after N tokens -> early stop")
    stop_after = 3
    collected: list[str] = []

    @TokenCallback
    def stop_early(piece, _user_data):
        collected.append(piece.decode("utf-8"))
        return len(collected) < stop_after

    rc = lib.llminference_complete_stream(
        handle, b"Say something.", c_uint32(100), c_float(0.0), stop_early, None
    )
    assert rc == OK, f"early-stop rc={rc}"
    print(f"    collected {len(collected)} tokens before stop (requested stop after {stop_after})")
    assert len(collected) == stop_after, f"expected {stop_after} tokens, got {len(collected)}"

    # Scenario 3: user_data round-trip (opaque pointer stays intact).
    print("  scenario 3: user_data round-trip through the callback")
    marker = ctypes.c_int(0xDEADBEEF)
    observed_ptr_value: list[int] = []

    @TokenCallback
    def check_user_data(_piece, user_data):
        observed_ptr_value.append(user_data or 0)
        return False  # one token is enough

    expected_ptr = ctypes.addressof(marker)
    rc = lib.llminference_complete_stream(
        handle,
        b"hi",
        c_uint32(1),
        c_float(0.0),
        check_user_data,
        ctypes.cast(ctypes.pointer(marker), c_void_p),
    )
    assert rc == OK, f"user_data rc={rc}"
    assert len(observed_ptr_value) == 1
    assert observed_ptr_value[0] == expected_ptr, (
        f"user_data mismatch: passed {expected_ptr:#x}, got {observed_ptr_value[0]:#x}"
    )
    print(f"    user_data passed={expected_ptr:#x} received={observed_ptr_value[0]:#x} OK")

    lib.llminference_destroy(handle)
    print("  destroy OK")
    print("\nStreaming smoke tests passed.")


if __name__ == "__main__":
    main()
