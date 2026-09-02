#!/bin/bash
#SBATCH --job-name=topo-scale-isobudget-deepseek
#SBATCH --account=gts-syu334-ece
#SBATCH --partition=cpu-medium
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/storage/scratch1/8/syoon351/repos/LLMServingSim/outputs/panel_dse/slurm_scale_isobudget_deepseek_%j.out
#SBATCH --error=/storage/scratch1/8/syoon351/repos/LLMServingSim/outputs/panel_dse/slurm_scale_isobudget_deepseek_%j.err

set -uo pipefail
export APPTAINER_CACHEDIR=/storage/home/hcoda1/8/syoon351/scratch/apptainer_cache
export APPTAINER_TMPDIR=/storage/home/hcoda1/8/syoon351/scratch/apptainer_tmp
REPO=/storage/scratch1/8/syoon351/repos/LLMServingSim
SIF=/storage/project/r-syu334-0/shared/images/astra-sim_latest.sif
VENV=/storage/home/hcoda1/8/syoon351/scratch/deps/llmservingsim_astra_venv

apptainer exec --cleanenv \
  --bind /storage/home/hcoda1/8/syoon351/scratch:/storage/home/hcoda1/8/syoon351/scratch \
  --bind /storage/scratch1/8/syoon351/repos:/storage/scratch1/8/syoon351/repos \
  --bind /storage/project/r-syu334-0:/storage/project/r-syu334-0 \
  --pwd "$REPO" "$SIF" bash -c "
    source $VENV/bin/activate
    export PIP_CACHE_DIR=/storage/home/hcoda1/8/syoon351/scratch/deps/pip_cache
    export HF_HOME=/storage/home/hcoda1/8/syoon351/scratch/hf_cache
    export TRANSFORMERS_CACHE=/storage/home/hcoda1/8/syoon351/scratch/hf_cache/transformers
    export HUGGINGFACE_HUB_CACHE=/storage/home/hcoda1/8/syoon351/scratch/hf_cache/hub
    cd /storage/scratch1/8/syoon351/repos && ln -sfn LLMServingSim/profiler profiler
    cd $REPO
    bash scripts/run_topo_scale_hpc.sh
  "
