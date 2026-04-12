#!/usr/bin/env python3
"""
Smoke test for libllmserv via ctypes.

Prereqs:
    - Build the cdylib:  cargo build --release --manifest-path llmserv/Cargo.toml -p llmserv-ffi
    - Ensure application.toml points at a model you have cached locally
      (or set XDG_CONFIG_HOME to a directory containing an override).

Tests (in order):
    1. init              — loads the model, returns a handle
    2. token_count       — fast tokenizer-only call
    3. tokenize          — returns token ids (Rust-allocated, freed via free_u32s)
    4. complete          — one generation round trip (slow, CPU-bound)
    5. destroy           — frees the handle

Exits non-zero on any failure. Prints a concise summary otherwise.
"""

import ctypes
import os
import sys
from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_uint32,
)

# ─── locate the built library ────────────────────────────────────────

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
TARGET_DIR = os.path.join(REPO_ROOT, "llmserv", "target", "release")

if sys.platform == "win32":
    LIB_NAME = "llmserv.dll"
elif sys.platform == "darwin":
    LIB_NAME = "libllmserv.dylib"
else:
    LIB_NAME = "libllmserv.so"

LIB_PATH = os.path.join(TARGET_DIR, LIB_NAME)
if not os.path.exists(LIB_PATH):
    print(f"ERR: {LIB_PATH} not found. Build first:", file=sys.stderr)
    print("    cargo build --release --manifest-path llmserv/Cargo.toml -p llmserv-ffi", file=sys.stderr)
    sys.exit(2)

lib = ctypes.CDLL(LIB_PATH)

# ─── bindings ────────────────────────────────────────────────────────

# LlmError codes (must match the Rust enum).
OK, INVALID_INPUT, LOAD_FAILED, RUNTIME, PANIC, INTERNAL = 0, 1, 2, 3, 4, 5

ERR_NAMES = {
    0: "OK",
    1: "INVALID_INPUT",
    2: "LOAD_FAILED",
    3: "RUNTIME",
    4: "PANIC",
    5: "INTERNAL",
}


class LlmHandle(ctypes.Structure):
    pass  # opaque


LlmHandlePtr = POINTER(LlmHandle)

lib.llmserv_init.argtypes = [POINTER(LlmHandlePtr)]
lib.llmserv_init.restype = c_int

lib.llmserv_destroy.argtypes = [LlmHandlePtr]
lib.llmserv_destroy.restype = None

lib.llmserv_complete.argtypes = [LlmHandlePtr, c_char_p, c_uint32, c_float, POINTER(c_char_p)]
lib.llmserv_complete.restype = c_int

lib.llmserv_embed.argtypes = [LlmHandlePtr, c_char_p, POINTER(POINTER(c_float)), POINTER(c_size_t)]
lib.llmserv_embed.restype = c_int

lib.llmserv_tokenize.argtypes = [LlmHandlePtr, c_char_p, POINTER(POINTER(c_uint32)), POINTER(c_size_t)]
lib.llmserv_tokenize.restype = c_int

lib.llmserv_token_count.argtypes = [LlmHandlePtr, c_char_p, POINTER(c_size_t)]
lib.llmserv_token_count.restype = c_int

lib.llmserv_free_string.argtypes = [c_char_p]
lib.llmserv_free_string.restype = None

lib.llmserv_free_floats.argtypes = [POINTER(c_float), c_size_t]
lib.llmserv_free_floats.restype = None

lib.llmserv_free_u32s.argtypes = [POINTER(c_uint32), c_size_t]
lib.llmserv_free_u32s.restype = None


def _check(code: int, fn: str) -> None:
    if code != OK:
        print(f"FAIL: {fn} returned {ERR_NAMES.get(code, code)}", file=sys.stderr)
        sys.exit(1)


# ─── tests ───────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading library: {LIB_PATH}")

    handle = LlmHandlePtr()
    _check(lib.llmserv_init(byref(handle)), "llmserv_init")
    print(f"  init OK (handle = {ctypes.addressof(handle.contents):#x})")

    # Fast path — tokenizer-only
    count = c_size_t()
    _check(lib.llmserv_token_count(handle, b"Hello, world!", byref(count)), "llmserv_token_count")
    print(f"  token_count('Hello, world!') = {count.value}")
    assert count.value > 0, "expected non-zero token count"

    # Tokenize — returns Rust-allocated buffer we have to free
    ids_ptr = POINTER(c_uint32)()
    ids_len = c_size_t()
    _check(
        lib.llmserv_tokenize(handle, b"Hello, world!", byref(ids_ptr), byref(ids_len)),
        "llmserv_tokenize",
    )
    ids = [ids_ptr[i] for i in range(ids_len.value)]
    lib.llmserv_free_u32s(ids_ptr, ids_len)
    print(f"  tokenize('Hello, world!') = {ids}")
    assert len(ids) == count.value, "tokenize and token_count disagree"

    # One completion — slow on CPU (~3-10s)
    print("  running completion (this takes a few seconds)...")
    out_text = c_char_p()
    _check(
        lib.llmserv_complete(handle, b"Say hello in 5 words.", c_uint32(12), c_float(0.0), byref(out_text)),
        "llmserv_complete",
    )
    text = out_text.value.decode("utf-8") if out_text.value else ""
    lib.llmserv_free_string(out_text)
    print(f"  complete -> {text!r}")
    assert text, "expected non-empty completion"

    lib.llmserv_destroy(handle)
    print("  destroy OK")
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
