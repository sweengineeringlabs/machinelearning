#!/usr/bin/env bash
set -euo pipefail

echo "Building swe-ml-architectures..."
cargo build -p swe-ml-architectures

echo "Running tests..."
cargo test -p swe-ml-architectures

echo "swe-ml-architectures bootstrap complete."
