#!/usr/bin/env python3
"""Decoupled prefill/decode-disaggregation fabric comparison.

Replaces the blocked single-simulation, two-dp_group approach (deadlocks in
ASTRA-Sim's C++ collective execution -- see LLMServingSim commit 217b7dc and
NSDI_2027_GlassPhotonics/README.md) with two INDEPENDENT single-pool
simulations (prefill-only, decode-only -- each already proven to run cleanly
up to EP=128, e.g. outputs/panel_dse/topo_prefill_*_isobudget.csv and
outputs/panel_dse/robust_test/deepseek_decode_ep*.csv) plus an ANALYTICAL
KV-transfer cost between them, using the exact same formula already
implemented in serving/core/router.py (transfer_prefill_request):

    transfer_ns = kv_bytes / kv_transfer_bw_GBps + kv_transfer_latency_ns

This mirrors the architecture of the separate PhotonicInference project
(~/Documents/vscode_workspace/PhotonicInference), which never asks ASTRA-Sim
to simulate the prefill->decode KV hop as a real collective either -- it is
computed as a closed-form Python cost (their router.py: circuit_stall_for_batch)
decoupled from the real network sim. We adopt the same decoupling, without
their OCS reconfiguration-delay machinery (our glass FB fabric is switchless,
delta = 0 by construction -- see ASP-DAC #362 / NSDI Sec. 4).

kv_bytes is computed by actually instantiating serving/core/memory_model.py's
MemoryModel and calling get_kv(seq) -- NOT hand-derived -- so it can never
drift from the simulator's own MLA KV-cache formula.

Usage (inside the apptainer/venv, from the LLMServingSim repo root):
    python scripts/pd_disagg_decoupled_analysis.py \
        --prefill-csv outputs/panel_dse/topo_prefill_deepseek_v3_0324_ep16_isobudget.csv \
                       outputs/panel_dse/topo_prefill_deepseek_v3_0324_ep32_isobudget.csv \
                       outputs/panel_dse/topo_prefill_deepseek_v3_0324_ep64_isobudget.csv \
        --isl 2048 \
        --out outputs/panel_dse/pd_disagg_decoupled_deepseek_v3_0324.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.core.memory_model import MemoryModel

# Same (name, link_bw GB/s, link_latency ns) triples as
# scripts/make_pd_disagg_configs.py::SCENARIOS -- kept as a literal copy
# (not an import) because that module executes config-writing code at
# import time; duplicating three numbers here is safer than importing side
# effects. If SCENARIOS there changes, update this too.
SCENARIOS = {
    "panel_glass": (512.0, 100.0),   # our intra-panel glass-photonic FB fabric
    "scaleout_ib": (50.0, 500.0),    # conventional IB-class scale-out fabric
}


def fb_prefill_step_ms(csv_path, topology_value, time_col):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("topology") == topology_value and row.get("status") == "ok":
                return int(row["ep"]), float(row[time_col])
    raise RuntimeError(f"no ok {topology_value} row in {csv_path}")


def kv_bytes_for(model_name, isl, dtype_bits=16):
    # Values beyond kv_lora_rank/qk_rope_head_dim/n_layer/kv_fp are unused by
    # get_kv(); npu_mem is set absurdly large purely to pass the constructor's
    # weight-fits-in-memory check for a 671B-param model on num_npus=1 (that
    # check is irrelevant to the KV-size formula we actually want).
    mm = MemoryModel(
        model=model_name, instance_id=0, node_id=0, num_npus=1, tp_size=1,
        npu_mem=1_000_000, cpu_mem=512, block_size=16, fp=dtype_bits,
        enable_prefix_caching=False, enable_prefix_sharing=False,
        prefix_pool=None, prefix_storage=None, ep_size=1,
    )
    return mm.get_kv(isl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-csv", nargs="+", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V3-0324")
    ap.add_argument("--isl", type=int, default=2048)
    ap.add_argument("--topology-value", default="fb_2d",
                     help="value of the CSV 'topology' column to match "
                          "(fb_2d for --sweep topo_compare CSVs, "
                          "glass_fb for --sweep batch CSVs)")
    ap.add_argument("--time-col", default="prefill_step_ms",
                     help="CSV column holding the prefill-phase time "
                          "(prefill_step_ms for topo_compare CSVs, "
                          "ttft_ms for --sweep batch CSVs at short ISL, "
                          "where TTFT IS the prefill-phase time)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kv_bytes = kv_bytes_for(args.model, args.isl)
    print(f"KV bytes/request @ ISL={args.isl} ({args.model}): "
          f"{kv_bytes/1e6:.2f} MB")

    rows = []
    for path in args.prefill_csv:
        ep, prefill_ms = fb_prefill_step_ms(path, args.topology_value, args.time_col)
        for scenario, (bw, lat) in SCENARIOS.items():
            transfer_ns = kv_bytes / bw + lat
            transfer_ms = transfer_ns / 1e6
            ttft_ms = prefill_ms + transfer_ms
            tax_pct = 100.0 * transfer_ms / prefill_ms
            rows.append(dict(
                ep=ep, scenario=scenario, kv_bytes=kv_bytes,
                prefill_step_ms=prefill_ms, kv_transfer_ms=transfer_ms,
                ttft_disagg_ms=ttft_ms, disagg_tax_pct=tax_pct,
            ))
            print(f"  EP={ep:4d} {scenario:12s} prefill={prefill_ms:8.2f}ms "
                  f"+ kv_transfer={transfer_ms:6.3f}ms -> TTFT={ttft_ms:8.2f}ms "
                  f"(tax {tax_pct:5.3f}%)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
