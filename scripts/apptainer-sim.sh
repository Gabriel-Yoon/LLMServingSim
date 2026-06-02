SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SCRATCH="/storage/home/hcoda1/8/syoon351/scratch"
IMAGE="/storage/project/r-syu334-0/syoon351/containers/llmservingsim.sif"
APPTAINER_HOME="$SCRATCH/deps/apptainer_home"

HOST_GCC="$(command -v gcc)"
HOST_GXX="$(command -v g++)"

export APPTAINER_CACHEDIR="$SCRATCH/apptainer_cache"
export APPTAINER_TMPDIR="$SCRATCH/apptainer_tmp"

export APPTAINERENV_PIP_CACHE_DIR="$SCRATCH/deps/pip_cache"
export APPTAINERENV_XDG_CACHE_HOME="$SCRATCH/hf_cache/xdg"
export APPTAINERENV_HF_HOME="$SCRATCH/hf_cache"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$SCRATCH/hf_cache/hub"
export APPTAINERENV_TRANSFORMERS_CACHE="$SCRATCH/hf_cache/transformers"

export APPTAINERENV_CC="$HOST_GCC"
export APPTAINERENV_CXX="$HOST_GXX"

apptainer exec \
  --cleanenv \
  --home "$APPTAINER_HOME":/home/syoon351 \
  --bind "$REPO_ROOT":/app/LLMServingSim \
  --bind "$SCRATCH":"$SCRATCH" \
  --bind /storage/project/r-syu334-0/syoon351:/storage/project/r-syu334-0/syoon351 \
  --bind /usr/local/pace-apps:/usr/local/pace-apps \
  --pwd /app/LLMServingSim \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail

    export CC="'"$HOST_GCC"'"
    export CXX="'"$HOST_GXX"'"
    export CFLAGS="-pthread"
    export CXXFLAGS="-pthread"
    export LDFLAGS="-pthread"

    VENV="$HOME/venvs/llmservingsim"
    source "$VENV/bin/activate"

    exec bash -i
  '