#!/bin/bash
source /storage/home/hcoda1/8/syoon351/scratch/deps/llmservingsim_astra_venv/bin/activate
cd /storage/scratch1/8/syoon351/repos/LLMServingSim
for name in colocated panel_glass scaleout_ib; do
  echo "=== $name ==="
  timeout 600 python -m serving \
    --cluster-config configs/cluster/pd_disagg_deepseek_ep16_${name}.json \
    --dtype bfloat16 --block-size 16 \
    --dataset workloads/example_trace.jsonl \
    --output outputs/pd_disagg_deepseek_ep16_${name}.csv \
    --num-req 16 --log-level WARNING
  echo "EXIT_CODE_${name}=$?"
done
