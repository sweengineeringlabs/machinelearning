#!/usr/bin/env bash
set -euo pipefail

echo "Checking Rust toolchain..."
rustc --version
cargo --version

echo "Building tensor-engine..."
cargo build -p rustml-core

echo "Running tests..."
cargo test -p rustml-core

echo "Done."
