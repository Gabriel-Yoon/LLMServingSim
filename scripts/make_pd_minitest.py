#!/usr/bin/env python3
"""Minimal PD (2 dp_group) isolation test -- v2, INTERLEAVED instance order.
NPU-to-dim mapping for dims=[tp, num_groups, dp_size] (group dim in the
middle): position in dim d = (id % (offset_d * size_d)) / offset_d, offset
accumulates as offset *= size for each earlier dim (GeneralComplexTopology.cc).
For dims=[1,2,2]: NPU0=(grp0,dp0) NPU1=(grp1,dp0) NPU2=(grp0,dp1) NPU3=(grp1,dp1)
-> P (grp0) = {NPU0,NPU2}, D (grp1) = {NPU1,NPU3} -> instances must be P,D,P,D,...
"""
import json, os, sys

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HARDWARE = "H100"
NPU_MEM = {"mem_size": 80, "mem_bw": 3350, "mem_latency": 0}

def make_instance(pd_type, dp_group, ep):
    return {"model_name": MODEL, "hardware": HARDWARE, "npu_mem": NPU_MEM,
            "num_npus": 1, "tp_size": 1, "ep_size": ep, "dp_group": dp_group,
            "pd_type": pd_type}

def make_config(ep, split_pd):
    if split_pd:
        # interleaved: P0,D0,P1,D1,...
        instances = []
        for _ in range(ep):
            instances.append(make_instance("prefill", "P", ep))
            instances.append(make_instance("decode", "D", ep))
    else:
        instances = [make_instance(None, "A", ep) for _ in range(ep)]
    return {
        "num_nodes": 1, "link_bw": 512.0, "link_latency": 100.0,
        "nodes": [{"num_instances": len(instances), "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
                   "instances": instances}],
    }

if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "cluster")
    for split, tag in [(False, "single"), (True, "pd")]:
        cfg = make_config(ep, split)
        path = os.path.join(outdir, f"minitest_ep{ep}_{tag}.json")
        json.dump(cfg, open(path, "w"), indent=2)
        print("wrote", path)
