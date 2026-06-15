# Legacy / superseded plot scripts

These plotting scripts read the **pre-`sweep_panel_dse.py`** data layout
(`outputs/dse_results.csv`, `outputs/dse_*_sweep.csv`, `outputs/dse_plots/`,
the old `sweep_128gpu.py` output). They are kept for reference only and are
**not** part of the current ASP-DAC 2027 reach pipeline.

Use the current scripts in `scripts/` instead — they read
`outputs/panel_dse/` (written by `sweep_panel_dse.py`) and plot the
**bug-immune `tpot_gt_ms`** metric (MODE of ASTRA per-iteration decode cycles).
The completion-timing metrics (`tpot_avg_ms` / `tpot_steady_ms`) are distorted
by the DP+EP `add_done` dummy-completion bug and must not be plotted.

| Legacy script | Superseded by (current) |
|---|---|
| `plot_128gpu.py`       | `plot_reach.py` / `plot_panel_dse.py` (epscale) |
| `plot_dse_results.py`  | `plot_panel_dse.py` |
| `plot_fb_dse.py`       | `plot_panel_dse.py` |
| `plot_quick_preview.py`| `plot_panel_dse.py` |
| `plot_wg_sweep.py`     | `plot_panel_dse.py` (intra WG sweep) |
| `plot_latency.py`      | `plot_panel_dse.py` (inter / latency) |
| `plot_tpot.py`         | `plot_exp1_ti.py` (Throughput–Interactivity) |
| `plot_topology.py`     | `plot_fb_topology.py` / `plot_dse_topology.py` |

## Current reach plots (in `scripts/`)
- `plot_reach.py`        — F3: TPOT-vs-EP crossover, glass-FB vs NVL72 (tpot_gt_ms)
- `plot_batch_exposure.py` — P6: batch×EP network-exposure heatmap + B* (exposed_frac, tpot_gt_ms)
- `plot_panel_dse.py`    — intra/inter WG-bandwidth sweep + EP-scaling cliff (tpot_gt_ms)
- `plot_fb_topology.py` / `plot_dse_topology.py` — interconnect topology figures
- `plot_exp1_ti.py`      — Exp 1 Throughput–Interactivity curve

All driven by `sweep_panel_dse.py` (`--sweep epscale|intra|inter|batch_x_ep`).
