# ml-sdk bootstrap script
# Installs dependencies and verifies build

Write-Host "Building ml-sdk..."
cargo build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running tests..."
cargo test
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "ml-sdk bootstrap complete."
