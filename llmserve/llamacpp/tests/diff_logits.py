#!/usr/bin/env python3
"""
Diff two LOGIT-format dumps produced by:
  llminference/.../bin/dump_logits_native.rs   (native_rust path)
  llminference/.../tests/dump_logits.rs        (llama_cpp path)

Both must dump the same fixed token sequence so we're comparing
forward-pass numerics, not tokenizer behavior.

Reports:
  - vocab size + sample count match check
  - top-1 (argmax) token from each side
  - top-10 indices from each side + overlap count
  - max absolute difference, mean absolute difference
  - position of max-diff
  - rough verdict
"""
import sys


def load(path):
    logits = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("LOGIT "):
                continue
            _, idx, val = line.split(maxsplit=2)
            logits[int(idx)] = float(val)
    return logits


def main():
    if len(sys.argv) != 3:
        print("usage: diff_logits.py NATIVE.txt LLAMACPP.txt", file=sys.stderr)
        sys.exit(2)

    a = load(sys.argv[1])
    b = load(sys.argv[2])

    print(f"native_rust  logits: {len(a)}")
    print(f"llama_cpp    logits: {len(b)}")
    if len(a) != len(b):
        print(f"!! vocab size mismatch ({len(a)} vs {len(b)})")
        sys.exit(1)

    n = len(a)
    a_list = [a[i] for i in range(n)]
    b_list = [b[i] for i in range(n)]

    # argmax (top-1)
    a_top = max(range(n), key=lambda i: a_list[i])
    b_top = max(range(n), key=lambda i: b_list[i])
    print(f"\nargmax token: native={a_top}  llama_cpp={b_top}  match={'YES' if a_top == b_top else 'NO'}")

    # top-10
    a_sorted = sorted(range(n), key=lambda i: a_list[i], reverse=True)[:10]
    b_sorted = sorted(range(n), key=lambda i: b_list[i], reverse=True)[:10]
    overlap = set(a_sorted) & set(b_sorted)
    print(f"top-10 native    : {a_sorted}")
    print(f"top-10 llama_cpp : {b_sorted}")
    print(f"top-10 overlap   : {len(overlap)}/10  ids: {sorted(overlap)}")

    # numerical diffs
    diffs = [a_list[i] - b_list[i] for i in range(n)]
    abs_diffs = [abs(d) for d in diffs]
    max_diff = max(abs_diffs)
    max_idx = abs_diffs.index(max_diff)
    mean_diff = sum(abs_diffs) / n

    print(f"\nmax |diff|       : {max_diff:.4f}  at idx {max_idx}")
    print(f"  native[{max_idx}] = {a_list[max_idx]:.4f}")
    print(f"  llama [{max_idx}] = {b_list[max_idx]:.4f}")
    print(f"mean|diff|       : {mean_diff:.4f}")

    # ranges
    a_min, a_max = min(a_list), max(a_list)
    b_min, b_max = min(b_list), max(b_list)
    print(f"\nrange native     : [{a_min:.2f}, {a_max:.2f}]  span {a_max - a_min:.2f}")
    print(f"range llama_cpp  : [{b_min:.2f}, {b_max:.2f}]  span {b_max - b_min:.2f}")

    # verdict
    print()
    if a_top == b_top and len(overlap) >= 8:
        print("VERDICT: Logits agree at top tokens. Bug is likely NOT in forward pass.")
    elif len(overlap) == 0:
        print("VERDICT: Logits completely disagree. Forward pass produces different distribution entirely.")
    elif len(overlap) <= 3:
        print("VERDICT: Logits substantially diverge. Likely numerical bug in forward pass affecting many tokens.")
    else:
        print(f"VERDICT: Logits partially agree ({len(overlap)}/10 top-10 overlap). Bug present but may be subtle.")


if __name__ == "__main__":
    main()
