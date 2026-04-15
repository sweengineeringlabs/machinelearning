#!/usr/bin/env bash
# Bootstrap entry point - delegates to scripts/ci/build.sh
exec "$(dirname "$0")/scripts/ci/build.sh" "$@"
