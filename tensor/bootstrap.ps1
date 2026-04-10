# Bootstrap script for tensor-engine
# Verifies toolchain and builds the crate

$ErrorActionPreference = "Stop"

Write-Host "Checking Rust toolchain..."
rustc --version
cargo --version

Write-Host "Building tensor-engine..."
cargo build -p rustml-core

Write-Host "Running tests..."
cargo test -p rustml-core

Write-Host "Done."
