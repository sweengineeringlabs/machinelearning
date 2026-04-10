#!/usr/bin/env bash
set -euo pipefail

echo "Building ml-sdk..."
cargo build

echo "Running tests..."
cargo test

echo "ml-sdk bootstrap complete."
