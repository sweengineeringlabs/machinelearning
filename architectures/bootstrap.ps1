# swe-ml-architectures bootstrap script
# Installs dependencies and verifies build

Write-Host "Building swe-ml-architectures..."
cargo build -p swe-ml-architectures
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running tests..."
cargo test -p swe-ml-architectures
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "swe-ml-architectures bootstrap complete."
