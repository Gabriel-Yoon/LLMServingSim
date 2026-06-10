#!/bin/bash
# Fetch HuggingFace config.json files for gated/private models and
# save them into configs/model/<org>/<model>.json so the profiler can
# load them without HF Hub access at profile time.
#
# Run this ONCE on HPC after:
#   huggingface-cli login        (or set HF_TOKEN env variable)
#
# Usage (from LLMServingSim repo root, inside vLLM Docker):
#   ./profiler/fetch_hf_configs.sh
#
# After this script finishes, verify that each JSON has the correct
# dimension fields, then run the corresponding profile script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Fetching HF model configs ==="

python3 - <<'EOF'
import json, os, shutil, sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

# Correct HuggingFace model IDs (org/repo) → local config path
MODELS = [
    # (hf_model_id,                                       local_path)
    ("openai/gpt-oss-120b",                               "configs/model/openai/gpt-oss-120b.json"),
    # Nemotron-3-Super-120B is a Mamba-2 + MoE hybrid (model_type: nemotron_h).
    # The LLMServingSim profiler does NOT currently support SSM/Mamba layers.
    # Fetching the config for reference only — do not profile until Mamba-2
    # support is added.
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",     "configs/model/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16.json"),
]

YAML_DIR = Path("profiler/models")
KNOWN_YAMLS = {p.stem for p in YAML_DIR.glob("*.yaml")}

for model_id, out_str in MODELS:
    out_path = Path(out_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFetching: {model_id}")
    try:
        tmp = hf_hub_download(repo_id=model_id, filename="config.json")
        shutil.copy(tmp, out_path)
        print(f"  Saved → {out_path}")

        with open(out_path) as f:
            cfg = json.load(f)

        model_type   = cfg.get("model_type", "UNKNOWN")
        arch         = cfg.get("architectures", ["UNKNOWN"])[0]
        hidden       = cfg.get("hidden_size", "?")
        layers       = cfg.get("num_hidden_layers", "?")
        heads        = cfg.get("num_attention_heads", "?")
        kv_heads     = cfg.get("num_key_value_heads", "?")
        intermediate = cfg.get("intermediate_size", "?")
        vocab        = cfg.get("vocab_size", "?")

        print(f"  model_type   : {model_type}")
        print(f"  architecture : {arch}")
        print(f"  hidden_size  : {hidden}")
        print(f"  num_layers   : {layers}")
        print(f"  heads (Q/KV) : {heads} / {kv_heads}")
        print(f"  intermediate : {intermediate}")
        print(f"  vocab_size   : {vocab}")

        if model_type in KNOWN_YAMLS:
            print(f"  ✓ profiler/models/{model_type}.yaml exists")
        else:
            print(f"  ✗ profiler/models/{model_type}.yaml MISSING — profiling not yet supported")

    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Make sure you ran: huggingface-cli login")
        print(f"  Or set HF_TOKEN environment variable.")

print("\n=== Done. Review configs above before running profile scripts. ===")
EOF
