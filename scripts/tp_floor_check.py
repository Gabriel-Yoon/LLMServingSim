#!/usr/bin/env python3
"""TP floor check: decode TPOT vs tensor-parallel degree for DeepSeek-V3-0324.

Single instance, tp_size = ep_size = N (dense sharded by TP/ALLREDUCE, experts by
EP/ALLTOALL), decode-heavy controlled workload. Reports the bug-immune
tpot_gt_ms (steady-decode MODE) + steady exposed_frac so we can see whether
sharding the dense/MLA compute brings TPOT under the 15 ms SLO.

Usage (in docker):
  python scripts/tp_floor_check.py --tp-list 1 2 4 --batch 16 --isl 512 --osl 128
"""
import argparse, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_panel_dse as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "deepseek-ai/DeepSeek-V3-0324"


def make_tp_config(tp, hardware, npu_mem_gb, link_bw, link_lat):
    """Single instance, tp_size=ep_size=tp. Dense uses TP (ALLREDUCE), MoE uses EP."""
    return {
        "num_nodes": 1,
        "link_bw": link_bw,
        "link_latency": link_lat,
        "nodes": [{
            "num_instances": 1,
            "cpu_mem": {"mem_size": 4096, "mem_bw": 256, "mem_latency": 0},
            "instances": [{
                "model_name": MODEL,
                "hardware": hardware,
                "npu_mem": {"mem_size": npu_mem_gb, "mem_bw": 3350, "mem_latency": 0},
                "num_npus": tp,
                "tp_size": tp,
                "ep_size": tp,
                "pd_type": None,
            }],
        }],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp-list", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--isl", type=int, default=512)
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--hardware", default="H100")
    ap.add_argument("--npu-mem-gb", type=int, default=2048)
    ap.add_argument("--link-bw", type=float, default=900.0)   # NVLink unidir
    ap.add_argument("--link-lat", type=float, default=500.0)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--out", default="outputs/panel_dse/tp_floor.csv")
    args = ap.parse_args()

    # decode-heavy controlled workload: `batch` requests, short ISL, long OSL.
    # make_workload writes the file itself and returns (relpath, n_requests).
    wl_rel, n_req = S.make_workload(ep=1, batch_per_inst=args.batch,
                                    mode="controlled", isl=args.isl, osl=args.osl)

    results = []
    for tp in args.tp_list:
        cfg = make_tp_config(tp, args.hardware, args.npu_mem_gb, args.link_bw, args.link_lat)
        cfg_path = os.path.join(REPO, "configs", "cluster", f"_tp_floor_tp{tp}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        out_csv = f"outputs/panel_dse/_tp_floor_tp{tp}.csv"
        cmd = [
            "python", "-m", "serving",
            "--cluster-config", os.path.relpath(cfg_path, REPO),
            "--dtype", "bfloat16", "--block-size", "16",
            "--dataset", wl_rel,
            "--output", out_csv,
            "--max-num-seqs", str(args.batch + 4),
            "--log-level", "INFO",
        ]
        print(f"\n===== TP={tp} =====\n  $ {' '.join(cmd)}", flush=True)
        try:
            p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                               timeout=args.timeout)
            log = p.stdout + p.stderr
        except subprocess.TimeoutExpired as e:
            log = (e.stdout or "") + (e.stderr or "")
            print(f"  TP={tp} TIMEOUT", flush=True)
        gt, ef = S.parse_steady_decode(log)
        slo = "PASS" if (gt is not None and gt < 15) else "FAIL"
        results.append((tp, gt, ef, slo))
        print(f"  TP={tp}: tpot_gt={gt} ms  exposed_steady={ef}  TPOT_SLO<15ms={slo}", flush=True)

    print("\n========== TP FLOOR SUMMARY ==========")
    print(f"{'TP':<5}{'tpot_gt_ms':<14}{'exposed':<10}{'TPOT_SLO<15ms'}")
    for tp, gt, ef, slo in results:
        gs = f"{gt:.1f}" if gt is not None else "?"
        es = f"{ef:.3f}" if ef is not None else "?"
        print(f"{tp:<5}{gs:<14}{es:<10}{slo}")
    os.makedirs(os.path.join(REPO, "outputs", "panel_dse"), exist_ok=True)
    with open(os.path.join(REPO, args.out), "w") as f:
        f.write("tp,tpot_gt_ms,exposed_frac,slo\n")
        for tp, gt, ef, slo in results:
            f.write(f"{tp},{gt},{ef},{slo}\n")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
