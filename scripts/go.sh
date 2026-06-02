#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

apptainer shell \
  --cleanenv \
  --bind "$SCRATCH:$SCRATCH" \
  --bind "$REPOS_ROOT:$REPOS_ROOT" \
  --bind /storage/project/r-syu334-0:/storage/project/r-syu334-0 \
  --pwd "$LLMSIM_REPO" \
  "$IMAGE"
