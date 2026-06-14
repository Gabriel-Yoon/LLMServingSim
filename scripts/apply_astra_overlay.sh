#!/bin/bash
# Apply the local ASTRA-Sim C++ changes (FlattenedButterfly electrical/optical
# split + degenerate-EP exit fix in main.cc) onto the nested astra-sim
# submodules, then rebuild.
#
# Why an overlay instead of a submodule push: the astra-sim submodules point at
# the shared casys-kaist upstream repos, and the FlattenedButterfly base was
# already an uncommitted submodule change, so a plain patch would double-apply.
# Overwriting the exact files is robust regardless of the submodule's current
# (uncommitted) state.
#
# Run from the repo root (or anywhere — it cd's to the repo root):
#   bash scripts/apply_astra_overlay.sh
#
# Files overwritten:
#   astra-sim (L1):           astra-sim/.../congestion_unaware/main.cc
#   astra-network-analytical: FlattenedButterfly.{cpp,h}, NetworkParser.{cpp,h},
#                             Helper.cpp, Type.h
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repo root
OV="astra_sim_overlay"
L2="astra-sim/extern/network_backend/analytical"

if [[ ! -d "$OV" ]]; then
    echo "ERROR: $OV not found (run from a checkout that has the overlay)." >&2
    exit 1
fi

echo "[overlay] copying L1 (astra-sim) files..."
cp -rv "$OV/L1/." "astra-sim/"

echo "[overlay] copying L2 (astra-network-analytical) files..."
cp -rv "$OV/L2/." "$L2/"

echo "[overlay] rebuilding ASTRA-Sim (incremental)..."
NUM_THREADS="${NUM_THREADS:-$(nproc 2>/dev/null || echo 8)}"
( cd astra-sim/build/astra_analytical/build && cmake --build . -j "$NUM_THREADS" )

echo "[overlay] done — FlattenedButterfly electrical/optical + main.cc fix applied & rebuilt."
