#!/usr/bin/env python3
"""Build prefill/decode-disaggregated cluster configs for the KV-cache-
transfer comparison: colocated (no P/D split at all -- the "no disaggregation
tax" reference) vs. disaggregated over our glass panel fabric vs.
disaggregated over a conventional (IB-class) scale-out network. Reuses this
branch's already-established H100-consistent constants (AGENTS.md: glass
intra-panel=512GB/s iso-per-link ablation/100ns latency; IB=50GB/s/500ns)
rather than inventing new ones.

NOTE (2026-09-01): link_bw drives BOTH the KV-transfer cost (router.py) AND
the actual EP AllToAll collective bandwidth inside each pool (config_builder
reuses the same cluster-level field for both) -- it cannot be 0 for a
disaggregated scenario without also killing the underlying network sim for
the real EP collective. "Colocated" is therefore modeled as a single pool
with NO pd_type split (matching the topology-sweep convention already
proven to work), not a PD split with link_bw=0.
"""
import json, os, sys

MODEL = "deepseek-ai/DeepSeek-V3-0324"
HARDWARE = "H100"
NPU_MEM = {"mem_size": 141, "mem_bw": 3350, "mem_latency": 0}

# name: (link_bw GB/s, link_latency ns, split_pd)
SCENARIOS = {
    "colocated":   (512.0, 100.0, False),  # single pool, no P/D split -- reference
    "panel_glass": (512.0, 100.0, True),   # disaggregated, our intra-panel fabric
    "scaleout_ib": (50.0,  500.0, True),   # disaggregated, conventional IB-class
}

def pool(ep, pd_type, dp_group):
    return [
        {"model_name": MODEL, "hardware": HARDWARE, "npu_mem": NPU_MEM,
         "num_npus": 1, "tp_size": 1, "ep_size": ep, "dp_group": dp_group,
         "pd_type": pd_type}
        for _ in range(ep)
    ]

def make_config(ep, link_bw, link_latency, split_pd):
    if split_pd:
        instances = pool(ep, "prefill", "P") + pool(ep, "decode", "D")
    else:
        instances = pool(ep, None, "A")
    return {
        "num_nodes": 1,
        "link_bw": link_bw,
        "link_latency": link_latency,
        "nodes": [{"num_instances": len(instances), "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
                   "instances": instances}],
    }

if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "cluster")
    for name, (bw, lat, split_pd) in SCENARIOS.items():
        cfg = make_config(ep, bw, lat, split_pd)
        path = os.path.join(outdir, f"pd_disagg_deepseek_ep{ep}_{name}.json")
        json.dump(cfg, open(path, "w"), indent=2)
        print("wrote", path)
