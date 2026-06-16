"""
Glass-panel intra-/inter-panel bandwidth DSE (ASP-DAC 2027).

Uses the heterogeneous FlattenedButterfly link model added in this branch:
  - electrical RDL    for adjacent tiles  (grid distance == 1)  — FIXED
  - optical waveguide for far same-row/col tiles                — SWEPT (intra)
  - inter-panel Ring  (glass-panel perimeter optical I/O)       — SWEPT (inter)

WG spec: 1 waveguide group = 32λ × 32 Gb/s = 1.024 Tb/s = 128 GB/s.

Two sweeps
----------
intra : single-panel (EP == panel_size). Fix electrical RDL, sweep wg_count
        (optical far-link bw = wg_count × 128). Question: "how many WGs to
        match NVL72 within one panel?"
inter : multi-panel (EP == panel_size × k). Fix intra at the chosen sweet-spot
        WG count, sweep inter_bw (panel egress). Question: "how much panel
        egress bandwidth to match NVL72?"

Baselines: NVL72 modeled at 1800 GB/s bidirectional NVLink (consistent with
the electrical RDL convention), one config per EP.

Workload modes
--------------
controlled : every request arrives at t=0 (N == EP, one per instance, no dummy
             waves). Isolates the network: reports steady-state decode TPOT
             (median inter-token interval, warmup tokens dropped). DEFAULT.
realistic  : staggered 1ms arrivals (matches sweep_paper.py); use to validate
             the controlled sweet-spot under a realistic arrival pattern.

Usage (inside Docker, from /app/LLMServingSim):
  python scripts/sweep_panel_dse.py --sweep intra --dry-run
  python scripts/sweep_panel_dse.py --sweep intra --panels 4x4 6x6_4c
  python scripts/sweep_panel_dse.py --sweep inter --fixed-wg 8 --panels 4x4
  python scripts/sweep_panel_dse.py --sweep intra --mode realistic
"""

import argparse
import collections
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_SIZE = 151936

# Loaded in main() from configs/model/<model>.json — used by analytical_estimates.
_MODEL_CFG = {}

# ─────────────────────────── Fixed hardware / model ───────────────────────────
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HARDWARE   = "H100"
NPU_MEM    = {"mem_size": 80, "mem_bw": 3350, "mem_latency": 0}
CPU_MEM    = {"mem_size": 1024, "mem_bw": 512, "mem_latency": 0}
TP         = 1     # tensor-parallel degree per EP rank (shards dense; ALLREDUCE on a
                   # prepended FullyConnected TP dim, EP all-to-all on the FB dims)
POWER      = False # when True, attach a node "power" spec so the simulator
                   # integrates real NPU+HBM+link energy (see _power_spec)

BLOCK_SIZE = 16
MAX_SEQS   = 128
MAX_TOKENS = 2048
ISL        = 512
OSL        = 64    # enough decode steps to reach steady state in controlled mode
WARMUP_TOK = 8     # ITL indices dropped before taking the steady-state median

# ─────────────────────────── Link model constants ─────────────────────────────
# All bandwidths are UNIDIRECTIONAL (ASTRA's analytical send() serializes a
# chunk over a link at this rate, one direction). N_WG is the per-direction
# waveguide count: a "x6 bundle" = 6 WG total = 3 TX + 3 RX = N_WG=3.
WG_BW          = 128.0    # GB/s per waveguide (32 λ × 32 Gb/s, unidirectional)
ELEC_BW        = 1800.0   # GB/s electrical RDL (adjacent tiles)  — FIXED
ELEC_LAT       = 100.0    # ns
INTRA_OPT_LAT  = 100.0    # ns  TeraPHY CPO (E/O+O/E ~10ns + WDM + coupling + margin)
INTER_LAT      = 500.0    # ns  inter-panel: edge transceiver (~50-200ns) + short
                          # co-located fiber (1-2m, ~5-10ns); 5x the in-package
                          # intra CPO. (5000ns would imply ~1km fiber — not co-located.)

# NVL72 baseline. Switch-hop removal gives the glass panel a ~10x latency
# advantage (1000 ns NVSwitch hop vs 100 ns CPO); since the MoE AllToAll is
# latency-bound this is a core strength, not a neutralized axis. Bandwidth is
# unidirectional: NVL72's 1800 GB/s is the bidirectional spec → 900 unidir,
# matching our N_WG×128 unidirectional convention.
NVL72_ELEC_BW = 900.0     # 1800 GB/s bidirectional → 900 unidirectional
NVL72_IB_BW   = 50.0      # inter-rack InfiniBand (only used for EP > 64)
NVL72_LAT     = 500.0     # ns  NVSwitch hardware hop. The ~1-2us figures often
                          # cited include software/kernel-launch overhead; the
                          # hardware one-way hop is a few hundred ns. H100
                          # NVSwitch collectives are bandwidth-dominated, so this
                          # latency is a secondary factor vs the BW / IB-cliff.
                          # glass CPO hop ~100 ns -> ~5x (not 10x) latency edge.
NVL72_SWITCH_POWER = 540.0  # W/rack  (36 NVSwitch × 15 W); glass = passive WG → 0

# ─────────────────────────── Sweep axes ───────────────────────────────────────
# N_WG = per-direction waveguides; intra optical BW = N_WG × 128 GB/s.
# The micro-bump cap is grid-dependent (compute_nwg_cap): 4x4 (degree 6) ~ 5,
# 6x6 (degree 10) ~ 3. The sweep spans both so each panel's cap line lands in
# range; N_WG beyond a panel's cap is the "aggressive" region on the plot.
WG_COUNTS_DEFAULT  = [1, 2, 3, 4, 5, 6, 8]
INTER_BW_DEFAULT   = [64, 128, 256, 512, 1024]

# panel_name -> (panel_rows, panel_cols)
PANELS = {
    "4x4":    (4, 4),   # 16 nodes
    "6x6_4c": (4, 8),   # 32 nodes (rectangular approx of 6×6 minus 4 corners)
    "6x6":    (6, 6),   # 36 nodes
}


def compute_nwg_cap(grid_rows, grid_cols, tile_area_mm2=1600, bump_density=1.083):
    """Per-direction N_WG cap from the micro-bump budget for an (rows x cols)
    FlattenedButterfly panel.

    1 WG = 32 lambda = 32 micro-bumps (single-ended).
    FlattenedButterfly degree = (rows-1) + (cols-1): every GPU links directly to
    all others in its row and column. Larger grids => higher degree => each pair
    gets thinner => lower per-pair N_WG cap.

    Returns degree, WG/GPU (both directions), WG/pair (both directions), and the
    conservative per-direction cap nwg_cap = floor(per-direction WG/pair).
    """
    bumps = tile_area_mm2 * bump_density        # 1600 * 1.083 = 1733
    wg_per_gpu = bumps / 32                      # ~54 WG/GPU (both directions)
    degree = (grid_rows - 1) + (grid_cols - 1)   # 6x6=10, 4x4=6
    wg_per_pair_total = wg_per_gpu / degree      # both directions
    nwg_per_dir = wg_per_pair_total / 2          # split TX/RX
    return {
        'degree': degree,
        'wg_per_gpu': wg_per_gpu,
        'wg_per_pair_total': wg_per_pair_total,
        'nwg_per_dir': nwg_per_dir,
        'nwg_cap': int(nwg_per_dir),   # floor (conservative cap)
    }


def _instance(ep):
    return {"model_name": MODEL_NAME, "hardware": HARDWARE, "npu_mem": NPU_MEM,
            "num_npus": TP, "tp_size": TP, "ep_size": ep, "dp_group": "A", "pd_type": None}


def _power_spec(fabric, num_npus):
    """Node power spec for the simulator's energy integrator (config_builder reads
    node["power"]; auto-fills npu.num_npus and dram.mem_size). FIRST-CUT values —
    cited, TUNE for the paper. Only `link` differs by fabric (the DSE variable);
    everything else is identical so the glass-vs-NVL72 delta is the fabric +
    completion-time effect, not param skew.

      NPU  : H100 SXM 700 W TDP (active), ~150 W idle [NVIDIA H100 datasheet].
      HBM  : 3.9 pJ/bit [HBM3 ~3.9-7 pJ/bit literature].
      link : glass = optical CPO, 1.15 pJ/bit [PanelScale Tbl1], PASSIVE fabric
             (no switch) but always-on laser/SerDes static ~tens W/GPU (CLAUDE.md:
             x5 WG ~ 59 W/GPU at full BW; use ~40 W/GPU baseline).
             NVL72 = NVLink electrical + ACTIVE NVSwitch 540 W/rack / 72 = 7.5 W/GPU
             + SerDes ~ 12 W/GPU; electrical e_bit ~2 pJ/bit (conservative).
    """
    npu = {HARDWARE: {"idle_power": 150.0, "active_power": 700.0,
                      "standby_power": 150.0, "standby_duration": 0.0}}
    cpu = {"idle_power": 200.0, "active_power": 400.0, "util": 0.1}
    dram = {"dimm_size": 16, "idle_power": 5.0, "energy_per_bit": 3.9}  # mem_size auto
    nic = {"idle_power": 15.0, "num_nics": 1}
    storage = {"idle_power": 5.0, "num_devices": 1}
    if fabric == "glass":
        link = {"idle_power": 40.0, "num_links": num_npus, "energy_per_bit": 1.15}
    else:  # nvl72
        link = {"idle_power": 12.0, "num_links": num_npus, "energy_per_bit": 2.0}
    return {"base_node_power": 0.0, "npu": npu, "cpu": cpu, "dram": dram,
            "link": link, "nic": nic, "storage": storage}


def _node(ep):
    return {"num_instances": ep, "cpu_mem": CPU_MEM, "instances": [_instance(ep) for _ in range(ep)]}


def make_panel_config(rows, cols, ep, wg_count, inter_bw, intra_lat=None, inter_lat=None):
    """fb_2d config with the electrical/optical split. inter_bw=0 → single panel.
    intra_lat / inter_lat override the default optical link latencies (ns) — used
    by the latency-radix sweep; default to the module constants."""
    cfg = {
        "num_nodes": 1,
        "topology_config": {
            "type": "fb_2d",
            "panel_rows": rows, "panel_cols": cols,
            "elec_bw": ELEC_BW, "elec_latency": ELEC_LAT,        # adjacent (fixed)
            "wg_count": wg_count, "wg_bw": WG_BW,                # optical far (swept)
            "intra_opt_latency": (INTRA_OPT_LAT if intra_lat is None else float(intra_lat)),
            "inter_bw": float(inter_bw),
            "inter_lat": (INTER_LAT if inter_lat is None else float(inter_lat)),  # inter-panel
        },
        "nodes": [_node(ep)],
    }
    if POWER:
        cfg["nodes"][0]["power"] = _power_spec("glass", ep * TP)
    return cfg


NVL72_RACK = 64    # NVLink domain size used as the rack boundary (largest pow2 <=72 dividing 256/384)

def make_nvl72_config(ep):
    """NVL72 baseline: a 64-GPU NVLink domain (1800 GB/s); beyond it, EP spans
    racks over InfiniBand (~50 GB/s). EP<=64 → flat NVLink; EP>64 → [64, ep/64]
    with the cross-rack dim at IB bandwidth. This inter-rack IB cliff is the
    weak point large models hit, since they need EP >> 64."""
    cfg = {
        "num_nodes": 1,
        "topology_config": {
            "type": "hierarchical_fb", "panel_size": NVL72_RACK, "tile_size": NVL72_RACK,
            "elec_bw": NVL72_ELEC_BW, "intra_opt_bw": NVL72_ELEC_BW, "inter_bw": NVL72_IB_BW,
            "elec_latency": NVL72_LAT, "intra_opt_latency": NVL72_LAT, "inter_latency": NVL72_LAT,
        },
        "nodes": [_node(ep)],
    }
    if POWER:
        cfg["nodes"][0]["power"] = _power_spec("nvl72", ep * TP)
    return cfg


# ─────────────────────────── Workload ─────────────────────────────────────────
def make_workload(ep, batch_per_inst, mode, isl, osl):
    """N = ep × batch_per_inst requests. With controlled (all arrive t=0) and a
    short ISL, the requests accumulate so that ~batch_per_inst of them decode
    concurrently on each instance — driving total_len (and thus the EXPOSED MoE
    all-to-all message) up. That is the regime where inter-GPU bandwidth
    (FlattenedButterfly vs NVL72) actually lands on the critical path."""
    rng = random.Random(0)
    n = ep * batch_per_inst
    lines = []
    for i in range(n):
        arrival = 0 if mode == "controlled" else i * 1_000_000
        lines.append(json.dumps({
            "input_toks": isl, "output_toks": osl, "arrival_time_ns": arrival,
            "input_tok_ids": [rng.randint(0, VOCAB_SIZE - 1) for _ in range(isl)],
            "output_tok_ids": [rng.randint(0, VOCAB_SIZE - 1) for _ in range(osl)],
        }))
    path = os.path.join(REPO_ROOT, "outputs", "panel_dse", "workloads",
                        f"wl_{mode}_ep{ep}_b{batch_per_inst}_isl{isl}_osl{osl}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return os.path.relpath(path, REPO_ROOT), n


# ─────────────────────────── Parse ────────────────────────────────────────────
def parse_metrics(out_csv_abs):
    """Return steady-state TPOT (median ITL past warmup) + avg TPOT + throughput."""
    if not os.path.exists(out_csv_abs):
        return None
    rows = list(csv.DictReader(open(out_csv_abs)))
    if not rows:
        return None
    steady, tpots, ttfts, e2es = [], [], [], []
    for r in rows:
        if r.get("TPOT"):
            tpots.append(float(r["TPOT"]) / 1e6)
        if r.get("TTFT"):
            ttfts.append(float(r["TTFT"]) / 1e6)
        if r.get("latency"):
            e2es.append(float(r["latency"]) / 1e6)
        try:
            itl = [v / 1e6 for v in eval(r["ITL"])]   # ns -> ms
            steady.extend(itl[WARMUP_TOK:])
        except Exception:
            pass
    if not tpots:
        return None
    tpot_avg = sum(tpots) / len(tpots)
    # Steady-state per-token cost = MEAN of pooled post-warmup inter-token
    # intervals (= total steady decode time / steady token count). Robust to
    # the lockstep ITL clustering that all-simultaneous arrival induces, where
    # a per-request median would collapse onto the clustered near-zero values.
    tpot_ss = (sum(steady) / len(steady)) if steady else tpot_avg
    return {
        "ttft_ms": (sum(ttfts) / len(ttfts)) if ttfts else 0.0,
        "e2e_latency_ms": (sum(e2es) / len(e2es)) if e2es else 0.0,
        "tpot_avg_ms": tpot_avg,
        "tpot_steady_ms": tpot_ss,
        "interactivity_tok_s_user": (1000.0 / tpot_ss) if tpot_ss > 0 else 0.0,
        "n_completed": len(rows),
    }


# ─────────────────────────── Exposed-communication parse ──────────────────────
_EXPOSED_RE = re.compile(
    r"NPU\[(\d+)\] iteration (\d+) finished, (\d+) cycles, "
    r"exposed communication (\d+) cycles")


def parse_exposed_log(text):
    """Pull ASTRA's per-NPU 'exposed communication' from the controller INFO log.

    The value is CUMULATIVE (curr_tick - accumulated gpu-op ticks), so the last
    iteration per NPU holds the run total. We average total/exposed across NPUs.
    Returns total_cycles, exposed_cycles, compute_cycles (= total - exposed,
    i.e. the overlapped GPU-op time), and exposed_frac (exposed / total).
    """
    last = {}  # sys -> (iteration, total_cycles, exposed_cycles)
    for m in _EXPOSED_RE.finditer(text or ""):
        sysid, it, cyc, exc = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        if sysid not in last or it > last[sysid][0]:
            last[sysid] = (it, cyc, exc)
    if not last:
        return None
    tot = [v[1] for v in last.values()]
    exp = [v[2] for v in last.values()]
    total_cycles = sum(tot) / len(tot)
    exposed_cycles = sum(exp) / len(exp)
    compute_cycles = max(0.0, total_cycles - exposed_cycles)
    return {
        "total_cycles": total_cycles,
        "exposed_cycles": exposed_cycles,
        "compute_cycles": compute_cycles,
        "exposed_frac": (exposed_cycles / total_cycles) if total_cycles > 0 else 0.0,
    }


_ITER_RE = re.compile(r"NPU\[(\d+)\] iteration (\d+) finished, (\d+) cycles")


def parse_steady_decode(text):
    """Ground-truth STEADY-STATE DECODE step from ASTRA's per-iteration controller
    log. Returns (tpot_gt_ms, exposed_frac) computed on the SAME steady-decode
    iterations, so the two are consistent (the old parse_exposed_log gave a
    CUMULATIVE run-average exposed_frac — prefill included — which did not match
    the steady-decode tpot_gt).

    Both 'C cycles' and 'exposed E cycles' are cumulative per NPU, so per-iteration
    total_delta = C[i]-C[i-1] and exposed_delta = E[i]-E[i-1]. The decode step
    repeats every iteration, so the MODE of total_delta (binned to 0.1 ms) is the
    steady decode step (tpot_gt); the exposed fraction is averaged over exactly the
    iterations that land in that mode bin (prefill / ramp iterations excluded)."""
    by_npu = {}  # npu -> {iter: (total_cyc, exposed_cyc)}
    for m in _EXPOSED_RE.finditer(text or ""):
        npu, it, cyc, exc = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        by_npu.setdefault(npu, {})[it] = (cyc, exc)
    pairs = []  # (total_delta, exposed_delta) per iteration
    for iters in by_npu.values():
        ks = sorted(iters)
        for a, b in zip(ks, ks[1:]):
            dt = iters[b][0] - iters[a][0]
            de = iters[b][1] - iters[a][1]
            if dt > 0:
                pairs.append((dt, de))
    if not pairs:
        return None, None
    binned = collections.Counter(round(dt / 1e5) / 10.0 for dt, _ in pairs)
    mode_ms = float(binned.most_common(1)[0][0])
    # exposed fraction over only the steady (mode-bin) iterations
    sel = [(dt, de) for dt, de in pairs if round(dt / 1e5) / 10.0 == mode_ms]
    sum_dt = sum(dt for dt, _ in sel)
    sum_de = sum(max(0, de) for _, de in sel)
    ef = (sum_de / sum_dt) if sum_dt > 0 else None
    return mode_ms, ef


def parse_steady_decode_cycle(text):
    """Back-compat shim: steady decode tpot_gt only."""
    gt, _ = parse_steady_decode(text)
    return gt


OVERLAP_EFFICIENCY = 0.8   # eta: fraction of the overlappable comm a real
                           # micro-batch pipeline actually hides. Ideal overlap
                           # (eta=1) hides all comm under compute -> exposed~0,
                           # which is unphysical: DeepEP/LMSYS large-EP DECODE
                           # leaves ~15-30% exposed (fill/drain bubbles, kernel
                           # gaps, M=4 finite depth, load imbalance). eta=0.8
                           # lands a compute-bound fabric at ~15-25% exposed,
                           # matching that reported range.

def apply_overlap(tpot_gt_ms, exposed_frac, eta=OVERLAP_EFFICIENCY):
    """Analytical micro-batch comp-comm overlap (DeepEP/LMSYS; AIC++ Eq. 1).

    The serial trace fully exposes the MoE all-to-all (dispatch/combine) because
    a batch's expert compute depends on its own dispatch. Real serving pipelines
    M micro-batches so one micro-batch's all-to-all (network) overlaps another's
    expert compute (NPU). Split the serial iteration into compute = (1-ef)*tpot
    and comm = ef*tpot; the pipeline hides eta*min(comm, compute):

      hidden          = eta * min(comm, compute)
      exposed_overlap = comm - hidden
      tpot_overlap    = compute + exposed_overlap = tpot*(1 - eta*min(ef, 1-ef))

    eta<1 keeps a realistic residual (no unphysical 0% exposed). When comm
    dominates (ef>>0.5, e.g. the NVL72 inter-rack IB cliff) only a small slice is
    hideable, so exposed stays high — the fabric is genuinely network-bound.
    """
    if tpot_gt_ms is None or exposed_frac is None or tpot_gt_ms <= 0:
        return tpot_gt_ms, exposed_frac
    ef = max(0.0, min(1.0, float(exposed_frac)))
    hidden = eta * min(ef, 1.0 - ef)           # fraction of tpot hidden
    tpot_ov = tpot_gt_ms * (1.0 - hidden)
    exp_ov = (ef - hidden) / (1.0 - hidden) if (1.0 - hidden) > 0 else ef
    return tpot_ov, max(0.0, exp_ov)


def analytical_estimates(model_cfg, ep, batch_per_inst, inter_bw, intra_bw, fp_bytes=2):
    """Analytical companions to the logged exposed metric (for interpretation only).

    all_to_all_us : raw per-iteration AG+RS time if fully serialized on the
                    inter-GPU link (dispatch local chunk + combine total buffer),
                    summed over MoE layers. This is the comm BEFORE compute overlap.
    weight_load_us: per-rank expert-weight HBM load time per iteration — the
                    decode bottleneck that determines how much of all_to_all_us
                    is hidden. experts shrink as 1/EP, so this falls with EP.
    """
    hidden = model_cfg["hidden_size"]
    n_layers = model_cfg["num_hidden_layers"]
    n_experts = model_cfg.get("num_local_experts") or model_cfg.get("n_routed_experts") or 0
    k = model_cfg.get("num_experts_per_tok", 1)
    moe_ffn = model_cfg.get("moe_intermediate_size", model_cfg.get("intermediate_size", hidden))
    mem_bw = NPU_MEM["mem_bw"]  # GB/s

    # decode total_len per instance ~ batch_per_inst (controlled accumulation).
    total_len = max(1, batch_per_inst)
    # AG dispatch local chunk + RS combine total buffer (mirrors trace_generator).
    eff_comm_tok = total_len  # dp_sum anchoring not applied in this single-instance estimate
    dispatch_bytes = max(1, eff_comm_tok // max(ep, 1)) * (hidden + n_experts) * fp_bytes
    combine_bytes = eff_comm_tok * hidden * fp_bytes
    link_bw_Bpns = max(inter_bw, 1e-9)  # GB/s == B/ns
    a2a_us_per_layer = (dispatch_bytes + combine_bytes) / link_bw_Bpns / 1000.0
    all_to_all_us = a2a_us_per_layer * n_layers

    # per-rank expert weights: experts_per_rank * (gate_up + down) bytes.
    experts_per_rank = max(1, n_experts // max(ep, 1))
    per_expert_bytes = 3 * hidden * moe_ffn * fp_bytes  # gate_up (2x) + down
    weight_bytes = experts_per_rank * per_expert_bytes
    weight_load_us = (weight_bytes / (mem_bw)) / 1000.0 * n_layers

    return {"all_to_all_us": all_to_all_us, "weight_load_us": weight_load_us}


# ─────────────────────────── Run one config ───────────────────────────────────
_POWER_TOTAL_RE = re.compile(r'Total energy consumption \(kJ\):\s+([0-9.]+)')
_POWER_COMP_RE = re.compile(r'([A-Za-z][A-Za-z ]*?) energy consumption \(J\):\s+([0-9.]+)')


def parse_power(text):
    """Pull the simulator's final energy summary (printed by PowerModel) into
    columns. Empty dict if power modeling was off. NPU includes HBM (weights are
    LOCAL -> folded into NPU active power), Link is the fabric-differentiated part."""
    out = {}
    m = _POWER_TOTAL_RE.search(text or "")
    if m:
        out["energy_total_kj"] = float(m.group(1))
    key_of = {"npu": "energy_npu_j", "cpu": "energy_cpu_j",
              "memory": "energy_dram_j", "link": "energy_link_j"}
    for cm in _POWER_COMP_RE.finditer(text or ""):
        k = key_of.get(cm.group(1).strip().lower())
        if k:
            out[k] = float(cm.group(2))
    return out


CSV_FIELDS = ["label", "sweep", "mode", "topology", "fabric", "panel", "ep",
              "per_device_batch", "wg_count", "intra_opt_bw", "inter_bw", "status",
              "ttft_ms", "e2e_latency_ms", "tpot_avg_ms", "tpot_steady_ms", "tpot_gt_ms",
              "tpot_gt_overlap_ms", "exposed_frac_overlap",
              "interactivity_tok_s_user", "n_completed",
              "total_cycles", "exposed_cycles", "compute_cycles", "exposed_frac",
              "all_to_all_us", "weight_load_us",
              "energy_total_kj", "energy_npu_j", "energy_cpu_j", "energy_dram_j", "energy_link_j",
              "elapsed_s", "error"]


def run_one(label, cfg, ep, meta, mode, isl, osl, max_tokens, batch_per_inst, out_dir, dry_run, timeout):
    # Per-run batch (batch sweeps vary it per config); fall back to the CLI value.
    batch_per_inst = int(meta.get("per_device_batch", batch_per_inst))
    meta.setdefault("per_device_batch", batch_per_inst)
    # fabric is the coarse FB-vs-NVL72 label (topology carries the panel name).
    meta.setdefault("fabric", "nvl72" if meta.get("topology") == "nvl72" else "glass_fb")

    cfg_path = os.path.join(out_dir, "configs", f"{label}.json")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    cfg_rel = os.path.relpath(cfg_path, REPO_ROOT)
    out_csv = os.path.join(out_dir, "runs", f"{label}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_rel = os.path.relpath(out_csv, REPO_ROOT)
    wl_rel, n_req = make_workload(ep, batch_per_inst, mode, isl, osl)

    # max_num_seqs must admit the requested per-device batch so that
    # batch_per_inst requests actually decode concurrently (drives total_len,
    # and thus the exposed MoE all-to-all message).
    max_seqs = max(MAX_SEQS, batch_per_inst)

    # Prefill is INCLUDED (no --skip-prefill): the prefill MoE all-gather /
    # reduce-scatter moves ~MB-scale messages, making the run bandwidth-bound.
    # --log-level INFO is required so the controller emits the per-iteration
    # 'exposed communication' lines we parse for the mechanism columns.
    cmd = ["python", "-m", "serving", "--cluster-config", cfg_rel, "--dtype", "bfloat16",
           "--block-size", str(BLOCK_SIZE), "--max-num-seqs", str(max_seqs),
           "--max-num-batched-tokens", str(max_tokens), "--dataset", wl_rel,
           "--output", out_rel, "--num-req", str(n_req), "--log-level", "INFO"]
    # SBI=1 enables dual-sub-batch interleaving (comp-comm overlap, DeepEP/LMSYS-style)
    # so the exposed-comm fraction reflects an overlap-optimized server, not the
    # serial worst case.
    if os.environ.get("SBI"):
        cmd.append("--enable-sub-batch-interleaving")

    if dry_run:
        print(f"  [dry] {label:38s} ep={ep:<4d} b={batch_per_inst:<4d} {meta}")
        return {"label": label, "status": "dry", **meta}

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=timeout)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            # collapse the traceback to a single CSV-safe line (last error line)
            err = proc.stderr.strip().replace("\n", " | ").replace(",", ";")[-300:]
            return {"label": label, "status": "error", "elapsed_s": elapsed,
                    "error": err, **meta}
        m = parse_metrics(out_csv) or {}
        _log = (proc.stdout or "") + (proc.stderr or "")
        exposed = parse_exposed_log(_log) or {}   # cumulative total/exposed (reference cols)
        gt, ef = parse_steady_decode(_log)         # steady-decode tpot_gt + consistent exposed_frac
        if gt is not None:
            m["tpot_gt_ms"] = gt
        if ef is not None:
            # use the STEADY-DECODE exposed (same iterations as tpot_gt), not the
            # cumulative run-average (which folds in prefill) — they must share a basis.
            exposed["exposed_frac"] = ef
        # Analytical comp-comm overlap (DeepEP/AIC++): hide the all-to-all behind
        # the other micro-batch's compute. Reported alongside the serial tpot_gt.
        if gt is not None and ef is not None:
            t_ov, ef_ov = apply_overlap(gt, ef)
            m["tpot_gt_overlap_ms"] = t_ov
            m["exposed_frac_overlap"] = ef_ov
        analyt = analytical_estimates(_MODEL_CFG, ep, batch_per_inst,
                                      float(meta.get("inter_bw") or NVL72_ELEC_BW),
                                      float(meta.get("intra_opt_bw") or NVL72_ELEC_BW))
        return {"label": label, "status": "ok", "elapsed_s": elapsed, "error": "",
                **meta, **m, **exposed, **analyt, **parse_power(_log)}
    except subprocess.TimeoutExpired:
        return {"label": label, "status": "timeout", "elapsed_s": timeout,
                "error": f"timeout>{timeout}s", **meta}


def append_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────── Build run lists ──────────────────────────────────
def build_intra_runs(panels, wg_counts, mode, isl, osl):
    runs = []
    for pname in panels:
        rows, cols = PANELS[pname]
        ep = rows * cols  # full single panel
        for wg in wg_counts:
            cfg = make_panel_config(rows, cols, ep, wg_count=wg, inter_bw=0.0)
            meta = {"sweep": "intra", "mode": mode, "topology": pname, "panel": ep,
                    "ep": ep, "wg_count": wg, "intra_opt_bw": int(wg * WG_BW), "inter_bw": 0}
            runs.append((f"intra_{pname}_ep{ep}_wg{wg}", cfg, ep, meta))
        # NVL72 baseline at this EP
        runs.append((f"intra_{pname}_ep{ep}_nvl72", make_nvl72_config(ep), ep,
                     {"sweep": "intra", "mode": mode, "topology": "nvl72", "panel": ep,
                      "ep": ep, "wg_count": 0, "intra_opt_bw": int(NVL72_ELEC_BW), "inter_bw": 0}))
    return runs


def build_epscale_runs(panel, wg, inter_opt_bw, ep_list, mode):
    """EP-scaling: sweep EP, compare glass-FB (multi-panel optical) vs NVL72
    (NVLink rack + inter-rack IB cliff). The large-model story — at EP >> 64
    NVL72 falls to IB while the glass panel's optical inter-panel BW holds."""
    rows, cols = panel
    panel_size = rows * cols
    runs = []
    for ep in ep_list:
        # glass FB: single panel if EP fits, else multi-panel over optical fiber
        multi = ep > panel_size
        cfg = make_panel_config(rows, cols, ep, wg_count=wg,
                                inter_bw=(inter_opt_bw if multi else 0.0))
        runs.append((f"epscale_fb_ep{ep}", cfg, ep,
                     {"sweep": "epscale", "mode": mode, "topology": "glass_fb", "panel": panel_size,
                      "ep": ep, "wg_count": wg, "intra_opt_bw": int(wg * WG_BW),
                      "inter_bw": (inter_opt_bw if multi else 0)}))
        runs.append((f"epscale_nvl72_ep{ep}", make_nvl72_config(ep), ep,
                     {"sweep": "epscale", "mode": mode, "topology": "nvl72", "panel": NVL72_RACK,
                      "ep": ep, "wg_count": 0, "intra_opt_bw": int(NVL72_ELEC_BW),
                      "inter_bw": (NVL72_IB_BW if ep > NVL72_RACK else 0)}))
    return runs


def build_inter_runs(panels, inter_bws, fixed_wg, mode, isl, osl):
    runs = []
    for pname in panels:
        rows, cols = PANELS[pname]
        panel = rows * cols
        for k in (2, 4):  # number of panels
            ep = panel * k
            if ep > 128:
                continue
            for ibw in inter_bws:
                cfg = make_panel_config(rows, cols, ep, wg_count=fixed_wg, inter_bw=ibw)
                meta = {"sweep": "inter", "mode": mode, "topology": pname, "panel": panel,
                        "ep": ep, "wg_count": fixed_wg, "intra_opt_bw": int(fixed_wg * WG_BW),
                        "inter_bw": ibw}
                runs.append((f"inter_{pname}_ep{ep}_wg{fixed_wg}_ib{ibw}", cfg, ep, meta))
            runs.append((f"inter_{pname}_ep{ep}_nvl72", make_nvl72_config(ep), ep,
                         {"sweep": "inter", "mode": mode, "topology": "nvl72", "panel": panel,
                          "ep": ep, "wg_count": 0, "intra_opt_bw": int(NVL72_ELEC_BW),
                          "inter_bw": NVL72_IB_BW}))
    return runs


def build_latency_runs(panel, wg, lat_list, axis, mode, inter_opt_bw):
    """Latency-radix sweep: hold BW (WG) and EP fixed, sweep the optical LINK
    LATENCY and watch decode TPOT / TTFT. At sufficient WG the collective is
    latency-bound (not BW-bound), so this isolates the glass switch-free hop
    (~100 ns intra / ~500 ns inter) vs the NVL72 switched hop (1000 ns).

      axis='intra' : single panel (EP=panel_size), sweep intra_opt_latency.
      axis='inter' : two panels  (EP=2*panel_size), sweep inter_lat.

    A single NVL72 run (fixed 1000 ns hop) is the horizontal reference.
    """
    rows, cols = panel
    panel_size = rows * cols
    ep = panel_size if axis == "intra" else panel_size * 2
    multi = ep > panel_size
    runs = []
    for lat in lat_list:
        if axis == "intra":
            cfg = make_panel_config(rows, cols, ep, wg_count=wg, inter_bw=0.0, intra_lat=lat)
        else:
            cfg = make_panel_config(rows, cols, ep, wg_count=wg,
                                    inter_bw=inter_opt_bw, inter_lat=lat)
        meta = {"sweep": "latency", "mode": mode, "topology": "glass_fb", "fabric": "glass_fb",
                "panel": panel_size, "ep": ep, "wg_count": wg, "lat_axis": axis,
                "link_lat": lat, "intra_opt_bw": int(wg * WG_BW),
                "inter_bw": (inter_opt_bw if multi else 0)}
        runs.append((f"latency_{axis}_ep{ep}_lat{int(lat)}", cfg, ep, meta))
    # NVL72 reference (fixed NVL72_LAT hop)
    runs.append((f"latency_{axis}_ep{ep}_nvl72", make_nvl72_config(ep), ep,
                 {"sweep": "latency", "mode": mode, "topology": "nvl72", "fabric": "nvl72",
                  "panel": NVL72_RACK, "ep": ep, "wg_count": 0, "lat_axis": axis,
                  "link_lat": NVL72_LAT, "intra_opt_bw": int(NVL72_ELEC_BW),
                  "inter_bw": (NVL72_IB_BW if ep > NVL72_RACK else 0)}))
    return runs


def _fabric_pair(sweep_name, panel, wg, inter_opt_bw, ep, batch, mode):
    """Build the FB-glass + NVL72 run pair for one (EP, per-device batch) cell.

    Both fabrics share EP, model and per-device batch; only the inter-GPU link
    differs (glass optical fiber vs NVLink-rack + inter-rack IB). This isolates
    'when does the EP all-to-all leave the compute shadow' as a function of
    batch (message size up) and EP (per-rank weight-load down)."""
    rows, cols = panel
    panel_size = rows * cols
    multi = ep > panel_size
    fb_cfg = make_panel_config(rows, cols, ep, wg_count=wg,
                               inter_bw=(inter_opt_bw if multi else 0.0))
    fb_meta = {"sweep": sweep_name, "mode": mode, "topology": "glass_fb", "fabric": "glass_fb",
               "panel": panel_size, "ep": ep, "per_device_batch": batch, "wg_count": wg,
               "intra_opt_bw": int(wg * WG_BW), "inter_bw": (inter_opt_bw if multi else 0)}
    nvl_meta = {"sweep": sweep_name, "mode": mode, "topology": "nvl72", "fabric": "nvl72",
                "panel": NVL72_RACK, "ep": ep, "per_device_batch": batch, "wg_count": 0,
                "intra_opt_bw": int(NVL72_ELEC_BW),
                "inter_bw": (NVL72_IB_BW if ep > NVL72_RACK else 0)}
    return [
        (f"{sweep_name}_fb_ep{ep}_b{batch}", fb_cfg, ep, fb_meta),
        (f"{sweep_name}_nvl72_ep{ep}_b{batch}", make_nvl72_config(ep), ep, nvl_meta),
    ]


def build_batch_runs(panel, wg, inter_opt_bw, ep, batch_list, mode):
    """Fixed EP, sweep per-device batch over both fabrics. Tests the hypothesis
    that exposed=0 at small batch flips to exposed>0 as the batch grows."""
    runs = []
    for batch in batch_list:
        runs += _fabric_pair("batch", panel, wg, inter_opt_bw, ep, batch, mode)
    return runs


def build_batch_x_ep_runs(panel, wg, inter_opt_bw, batch_list, ep_list, mode):
    """Full batch x EP grid over both fabrics — the data behind the P6 exposure
    heatmap and the B*(EP) crossover table."""
    runs = []
    for ep in ep_list:
        for batch in batch_list:
            runs += _fabric_pair("batch_x_ep", panel, wg, inter_opt_bw, ep, batch, mode)
    return runs


def main():
    global MODEL_NAME, HARDWARE, NVL72_RACK, INTER_LAT, TP, POWER
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", choices=["intra", "inter", "epscale", "batch", "batch_x_ep", "latency"],
                    required=True)
    ap.add_argument("--lat-list", nargs="+", type=float, default=[30, 100, 300, 500, 1000],
                    help="latency sweep: optical link latencies in ns (glass ~100 intra/500 inter "
                         "vs NVL72 1000 hop)")
    ap.add_argument("--lat-axis", choices=["intra", "inter"], default="intra",
                    help="latency sweep: which optical link latency to vary (intra single-panel "
                         "or inter multi-panel)")
    ap.add_argument("--ep-list", nargs="+", type=int, default=[16, 32, 64, 128, 256],
                    help="epscale/batch_x_ep: EP degrees to sweep (must divide model experts)")
    ap.add_argument("--batch-ep", type=int, default=64,
                    help="batch sweep: the single fixed EP to hold while sweeping per-device batch")
    ap.add_argument("--batch-list", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256, 512],
                    help="batch / batch_x_ep: per-device batch sizes to sweep")
    ap.add_argument("--epscale-panel", nargs=2, type=int, default=[4, 8],
                    help="epscale: glass panel (rows cols), default 4x8=32 (6x6-4c)")
    ap.add_argument("--inter-opt-bw", type=int, default=512,
                    help="epscale: glass inter-panel optical egress BW (GB/s)")
    ap.add_argument("--inter-lat", type=float, default=INTER_LAT,
                    help="glass inter-panel fiber latency (ns). Default 500 (co-located: "
                         "edge transceiver + short fiber); <1%% TPOT effect (BW-bound).")
    ap.add_argument("--nvl72-rack", type=int, default=NVL72_RACK,
                    help="NVLink domain (rack) size used as the cross-rack boundary. "
                         "Lower it (e.g. 4) so a small EP crosses into the inter-rack IB "
                         "dim — lets the cliff be reproduced locally without 128 instances.")
    ap.add_argument("--mode", choices=["controlled", "realistic"], default="controlled")
    ap.add_argument("--panels", nargs="+", default=["4x4", "6x6_4c", "6x6"], choices=list(PANELS))
    ap.add_argument("--wg", nargs="+", type=int, default=WG_COUNTS_DEFAULT, help="intra: WG counts")
    ap.add_argument("--inter-bw", nargs="+", type=int, default=INTER_BW_DEFAULT, help="inter: egress BW (GB/s)")
    ap.add_argument("--fixed-wg", type=int, default=8, help="inter: intra optical WG count to hold fixed")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="HF model id (must have configs/model/<id>.json and profiler/perf/<hw>/<id>/). "
                         "DeepSeek-V3 (hidden 7168, 256 experts, 61 layers) is far more bandwidth-bound than Qwen.")
    ap.add_argument("--hardware", default=HARDWARE)
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor-parallel degree per EP rank (shards dense compute; "
                         "needs a profiled tp<N> folder). Total GPUs = TP x EP.")
    ap.add_argument("--power", action="store_true",
                    help="attach a node power spec so the simulator integrates real "
                         "NPU+HBM+link energy (glass optical vs NVL72 NVLink+NVSwitch).")
    ap.add_argument("--npu-mem-gb", type=int, default=NPU_MEM["mem_size"],
                    help="per-NPU memory capacity (GB). Raise for large models like DeepSeek-V3 at tp=1 "
                         "(671B replicates dense weights per GPU); only gates the weight-fit/KV check, "
                         "not network timing.")
    ap.add_argument("--batch-per-instance", type=int, default=1,
                    help="concurrent requests per instance (N = EP × this). >1 raises decode total_len "
                         "so the EP all-to-all message is exposed — the regime where FB bandwidth matters.")
    ap.add_argument("--isl", type=int, default=ISL)
    ap.add_argument("--osl", type=int, default=OSL)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="prefill batch size (tokens/step). Larger → larger MoE messages → more bandwidth-bound.")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None, help="results CSV (default outputs/panel_dse/dse_<sweep>_<mode>.csv)")
    args = ap.parse_args()

    # Override the model/hardware globals used by the config builders.
    MODEL_NAME = args.model
    HARDWARE = args.hardware
    TP = args.tp
    POWER = args.power
    NPU_MEM["mem_size"] = args.npu_mem_gb
    NVL72_RACK = args.nvl72_rack
    INTER_LAT = args.inter_lat
    global _MODEL_CFG
    with open(os.path.join(REPO_ROOT, "configs", "model", MODEL_NAME + ".json")) as f:
        _MODEL_CFG = json.load(f)

    out_dir = os.path.join(REPO_ROOT, "outputs", "panel_dse")
    results_csv = args.out or os.path.join(out_dir, f"dse_{args.sweep}_{args.mode}.csv")
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    if args.sweep == "intra":
        runs = build_intra_runs(args.panels, args.wg, args.mode, args.isl, args.osl)
    elif args.sweep == "epscale":
        runs = build_epscale_runs(tuple(args.epscale_panel), args.fixed_wg, args.inter_opt_bw,
                                  args.ep_list, args.mode)
    elif args.sweep == "batch":
        runs = build_batch_runs(tuple(args.epscale_panel), args.fixed_wg, args.inter_opt_bw,
                                args.batch_ep, args.batch_list, args.mode)
    elif args.sweep == "batch_x_ep":
        runs = build_batch_x_ep_runs(tuple(args.epscale_panel), args.fixed_wg, args.inter_opt_bw,
                                     args.batch_list, args.ep_list, args.mode)
    elif args.sweep == "latency":
        runs = build_latency_runs(tuple(args.epscale_panel), args.fixed_wg, args.lat_list,
                                  args.lat_axis, args.mode, args.inter_opt_bw)
    else:
        runs = build_inter_runs(args.panels, args.inter_bw, args.fixed_wg, args.mode, args.isl, args.osl)

    print(f"\n{'='*72}")
    print(f"Panel DSE — sweep={args.sweep}  mode={args.mode}  model={MODEL_NAME} hw={HARDWARE}")
    print(f"  electrical RDL (fixed): {ELEC_BW} GB/s / {ELEC_LAT} ns")
    if args.sweep == "intra":
        print(f"  WG counts: {args.wg}  (optical = WG×{WG_BW:.0f} GB/s)")
    elif args.sweep == "epscale":
        print(f"  EP list: {args.ep_list}  glass panel {args.epscale_panel} wg{args.fixed_wg}, "
              f"inter-panel optical {args.inter_opt_bw} GB/s  vs  NVL72 (NVLink {NVL72_ELEC_BW} / "
              f"inter-rack IB {NVL72_IB_BW} @EP>{NVL72_RACK})")
    elif args.sweep == "batch":
        print(f"  fixed EP: {args.batch_ep}  per-device batch sweep: {args.batch_list}  "
              f"glass panel {args.epscale_panel} wg{args.fixed_wg} inter-opt {args.inter_opt_bw}  vs  "
              f"NVL72 (IB {NVL72_IB_BW} @EP>{NVL72_RACK})")
    elif args.sweep == "batch_x_ep":
        print(f"  EP x batch grid: {args.ep_list} x {args.batch_list}  "
              f"glass panel {args.epscale_panel} wg{args.fixed_wg} inter-opt {args.inter_opt_bw}  vs  "
              f"NVL72 (IB {NVL72_IB_BW} @EP>{NVL72_RACK})")
    elif args.sweep == "latency":
        print(f"  latency-radix ({args.lat_axis}): link latency sweep {args.lat_list} ns  "
              f"glass panel {args.epscale_panel} wg{args.fixed_wg}  vs  NVL72 ({NVL72_LAT:.0f} ns hop)")
    else:
        print(f"  fixed intra WG: {args.fixed_wg} (={int(args.fixed_wg*WG_BW)} GB/s)  inter_bw sweep: {args.inter_bw}")
    _wl_b = "per-run" if args.sweep in ("batch", "batch_x_ep") else args.batch_per_instance
    print(f"  workload: N=EP×{_wl_b}, ISL/OSL {args.isl}/{args.osl}, "
          f"max_tokens={args.max_tokens}, arrivals={'t=0' if args.mode=='controlled' else '1ms staggered'}")
    print(f"  baseline: NVL72 {NVL72_ELEC_BW} GB/s unidirectional, {NVL72_LAT:.0f} ns hop")
    print(f"  runs: {len(runs)}   results: {results_csv}")
    print(f"{'='*72}\n")

    for label, cfg, ep, meta in runs:
        row = run_one(label, cfg, ep, meta, args.mode, args.isl, args.osl, args.max_tokens,
                      args.batch_per_instance, out_dir, args.dry_run, args.timeout)
        if not args.dry_run:
            append_row(results_csv, row)
            tag = row.get("status")
            if row.get("ttft_ms") is not None and tag == "ok":
                print(f"  {label:44s} {tag:6s} "
                      f"TTFT={row['ttft_ms']:8.2f}ms  TPOT_gt={row.get('tpot_gt_ms', 0) or 0:8.3f}ms  "
                      f"exposed={row.get('exposed_frac', 0)*100:5.1f}%")
            else:
                print(f"  {label:40s} {tag}  {row.get('error','')[:80]}")

    print("\n=== done ===" + ("  (dry-run)" if args.dry_run else f"  → {results_csv}"))


if __name__ == "__main__":
    main()
