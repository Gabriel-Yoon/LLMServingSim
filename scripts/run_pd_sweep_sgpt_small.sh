#!/bin/bash
source /storage/home/hcoda1/8/syoon351/scratch/deps/llmservingsim_astra_venv/bin/activate
cd /storage/scratch1/8/syoon351/repos/LLMServingSim
for name in colocated panel_glass scaleout_ib; do
  echo "=== $name ==="
  timeout 3000 python -m serving \
    --cluster-config configs/cluster/pd_disagg_deepseek_ep16_${name}.json \
    --dtype bfloat16 --block-size 16 \
    --dataset workloads/sharegpt-deepseek-1000.jsonl \
    --output outputs/pd_disagg_deepseek_ep16_${name}_sgpt8.csv \
    --num-req 8 --log-level WARNING
  echo "EXIT_CODE_${name}=$?"
done
