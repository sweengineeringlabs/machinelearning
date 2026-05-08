#!/usr/bin/env python3
"""
Destroy-safety smoke test. Verifies that every "Not safe" scenario
from the original thread-safety table is now handled without UB:

    1. Double-destroy returns without crashing (idempotent no-op)
    2. Use-after-destroy returns LlmError::Destroyed (= 6), not a segfault
    3. Destroy concurrent with in-flight call serializes cleanly: the
       in-flight call completes with OK, the destroy waits for it, and
       subsequent calls return Destroyed

If any of these scenarios crashed the process, this script would never
reach the "PASS" line — it would hard-exit.
"""

import ctypes
import os
import sys
import threading
import time
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_size_t, c_uint32

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
TARGET_DIR = os.path.join(REPO_ROOT, "llminference", "target", "release")
LIB_NAME = {"win32": "llminference.dll", "darwin": "libllminference.dylib"}.get(sys.platform, "libllminference.so")
LIB_PATH = os.path.join(TARGET_DIR, LIB_NAME)

if not os.path.exists(LIB_PATH):
    print(f"ERR: {LIB_PATH} not found. Build first.", file=sys.stderr)
    sys.exit(2)

lib = ctypes.CDLL(LIB_PATH)

OK, INVALID_INPUT, LOAD_FAILED, RUNTIME, PANIC, INTERNAL, DESTROYED = 0, 1, 2, 3, 4, 5, 6


class LlmHandle(ctypes.Structure):
    pass


LlmHandlePtr = POINTER(LlmHandle)

lib.llminference_init.argtypes = [POINTER(LlmHandlePtr)]
lib.llminference_init.restype = c_int
lib.llminference_destroy.argtypes = [LlmHandlePtr]
lib.llminference_destroy.restype = None
lib.llminference_token_count.argtypes = [LlmHandlePtr, c_char_p, POINTER(c_size_t)]
lib.llminference_token_count.restype = c_int
lib.llminference_complete.argtypes = [LlmHandlePtr, c_char_p, c_uint32, c_float, POINTER(c_char_p)]
lib.llminference_complete.restype = c_int
lib.llminference_free_string.argtypes = [c_char_p]
lib.llminference_free_string.restype = None


def count(h, text):
    n = c_size_t()
    return lib.llminference_token_count(h, text, byref(n)), n.value


def main() -> None:
    print(f"Loading library: {LIB_PATH}")
    handle = LlmHandlePtr()
    rc = lib.llminference_init(byref(handle))
    assert rc == OK, f"init failed rc={rc}"
    print("  init OK")

    # Baseline — calls work before destroy.
    rc, n = count(handle, b"Hello, world!")
    assert rc == OK, f"pre-destroy call rc={rc}"
    assert n > 0
    print(f"  pre-destroy token_count = {n} (OK)")

    # Scenario 1: double-destroy is a safe no-op.
    print("  scenario 1: double-destroy")
    lib.llminference_destroy(handle)    # first destroy
    lib.llminference_destroy(handle)    # second destroy — must not crash
    lib.llminference_destroy(handle)    # third destroy — must not crash
    print("    OK — no crash on repeated destroy")

    # Scenario 2: use-after-destroy returns Destroyed, not a segfault.
    print("  scenario 2: use-after-destroy")
    rc, _ = count(handle, b"Hello, world!")
    assert rc == DESTROYED, f"expected DESTROYED (6), got {rc}"
    print(f"    OK — token_count returned DESTROYED ({rc})")

    out = c_char_p()
    rc = lib.llminference_complete(handle, b"hi", c_uint32(4), c_float(0.0), byref(out))
    assert rc == DESTROYED, f"expected DESTROYED (6), got {rc}"
    if out.value:
        lib.llminference_free_string(out)
    print(f"    OK — complete returned DESTROYED ({rc})")

    # Scenario 3: destroy concurrent with in-flight call.
    # Use a fresh handle since the previous is already destroyed.
    print("  scenario 3: destroy concurrent with in-flight call")
    handle2 = LlmHandlePtr()
    rc = lib.llminference_init(byref(handle2))
    assert rc == OK
    print("    init OK")

    # Fire a slow operation (a completion) in a background thread, then
    # destroy from main thread while it's in flight. Destroy must wait.
    inflight_rc = []

    def slow_call():
        out_text = c_char_p()
        rc = lib.llminference_complete(
            handle2, b"Say hello briefly.", c_uint32(20), c_float(0.0), byref(out_text)
        )
        inflight_rc.append(rc)
        if out_text.value:
            lib.llminference_free_string(out_text)

    t = threading.Thread(target=slow_call)
    t.start()
    time.sleep(0.3)  # let the call get into the model forward pass

    # Destroy while the call is in flight. This must wait for it.
    print("    requesting destroy during in-flight completion...")
    destroy_started = time.time()
    lib.llminference_destroy(handle2)
    destroy_took = time.time() - destroy_started
    t.join()
    print(f"    destroy waited {destroy_took:.2f}s for in-flight call")

    # The in-flight call must have completed successfully (it started before destroy).
    assert len(inflight_rc) == 1
    assert inflight_rc[0] == OK, f"in-flight call rc={inflight_rc[0]}, expected OK"
    print(f"    in-flight call returned OK (destroy waited, didn't interrupt)")

    # Call after destroy must return Destroyed.
    rc, _ = count(handle2, b"Hello")
    assert rc == DESTROYED, f"post-destroy call rc={rc}, expected DESTROYED"
    print(f"    post-destroy call returned DESTROYED ({rc})")

    print("\nDestroy-safety contract verified end-to-end. All scenarios PASS.")


if __name__ == "__main__":
    main()
