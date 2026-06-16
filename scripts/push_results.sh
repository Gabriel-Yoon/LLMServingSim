#!/usr/bin/env bash
# Bring HPC result CSVs into git so they can be pulled to local.
#
# The result CSVs live under outputs/ which is .gitignored (generated data), so
# this force-adds ONLY the specific result files by explicit path. It NEVER runs
# `git add -A` and never touches secrets / the astra-sim submodule / large files.
# Review the staged list it prints before it commits+pushes.
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="research/asp-dac-2027"

# explicit result paths only (globs that match nothing are skipped)
PATHS=(
  outputs/panel_dse/reach_*.csv          # H2 reach (wg5 + wg4)
  outputs/slo_eval/*.csv                 # H1/H3 SLO goodput
)

staged=0
for p in "${PATHS[@]}"; do
  for f in $p; do
    [ -f "$f" ] || continue
    git add -f "$f" && staged=1
  done
done

if [ "$staged" -eq 0 ]; then
  echo "No result CSVs found to add."; exit 0
fi

echo "=== staged result files (these only) ==="
git status --short -- outputs/panel_dse/reach_*.csv outputs/slo_eval/*.csv 2>/dev/null || git status --short
echo "========================================="
git commit -m "Add HPC reach/SLO result CSVs (headline data)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push origin "$BRANCH"
echo "Pushed. On local: git pull"
