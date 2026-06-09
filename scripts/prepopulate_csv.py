"""
Pre-populate the main sweep results CSV from existing individual sim output CSVs.

Individual sim CSVs in outputs/sim_128gpu/ contain per-request data (ns units).
This script parses them and appends entries to the main dse CSV.

Usage:
    python scripts/prepopulate_csv.py \
        --sim-dir outputs/sim_128gpu \
        --results-csv outputs/dse_128gpu_4x4_vs_6x6.csv
"""

import argparse
import csv
import os
import re
import sys

WG_BW = 128
NVL72_ELEC_BW = 1800.0
NVL72_LAT = 1000.0

FIELDS = ["label", "status", "topology", "panel", "ep_nominal", "ep", "wg_count",
          "intra_opt_bw", "ttft_ms", "tpot_ms", "lat_ms", "ttft_p50_ms", "tpot_p50_ms",
          "n_reqs", "elapsed_s", "stderr", "error"]


def _parse_sim_csv(path):
    ttfts, tpots, lats = [], [], []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                ttfts.append(float(row["TTFT"]) / 1e6)
                tpots.append(float(row["TPOT"]) / 1e6)
                lats.append(float(row["latency"]) / 1e6)
    except Exception as e:
        return None, str(e)
    if not ttfts:
        return None, "empty"
    n = len(ttfts)
    return {
        "ttft_ms":     sum(ttfts) / n,
        "tpot_ms":     sum(tpots) / n,
        "lat_ms":      sum(lats)  / n,
        "ttft_p50_ms": sorted(ttfts)[n // 2],
        "tpot_p50_ms": sorted(tpots)[n // 2],
        "n_reqs":      n,
        "elapsed_s":   0,
    }, None


def _parse_label(label):
    # nvl72_ep16
    m = re.match(r"nvl72_ep(\d+)$", label)
    if m:
        ep = int(m.group(1))
        return {"topology": "nvl72", "panel": "nvl72",
                "ep_nominal": ep, "ep": ep,
                "wg_count": 14, "intra_opt_bw": NVL72_ELEC_BW}
    # fb_6x6_4c_ep16n_ep16_wg6_bw768
    m = re.match(r"fb_(\w+)_ep(\d+)n_ep(\d+)_wg(\d+)_bw(\d+)$", label)
    if m:
        return {"topology": "fb", "panel": m.group(1),
                "ep_nominal": int(m.group(2)), "ep": int(m.group(3)),
                "wg_count": int(m.group(4)), "intra_opt_bw": int(m.group(5))}
    return None


def label_from_filename(fname):
    """Convert filename like nvl72_ep16.csv → nvl72_ep16"""
    name = os.path.splitext(fname)[0]
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-dir",     default="outputs/sim_128gpu")
    parser.add_argument("--results-csv", default="outputs/dse_128gpu_4x4_vs_6x6.csv")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Overwrite existing entries in the results CSV")
    args = parser.parse_args()

    # Load existing main CSV
    done_labels = {}
    existing_rows = []
    if os.path.exists(args.results_csv):
        with open(args.results_csv) as f:
            for row in csv.DictReader(f):
                lbl = row["label"]
                done_labels[lbl] = row
                existing_rows.append(row)
        print(f"Loaded {len(existing_rows)} existing rows from {args.results_csv}")

    # Scan sim_dir for individual sim CSVs
    if not os.path.isdir(args.sim_dir):
        print(f"ERROR: sim-dir '{args.sim_dir}' not found")
        sys.exit(1)

    new_rows = []
    skipped = []
    errors = []

    for fname in sorted(os.listdir(args.sim_dir)):
        if not fname.endswith(".csv"):
            continue
        label = label_from_filename(fname)
        meta = _parse_label(label)
        if meta is None:
            continue  # doesn't match expected patterns

        if label in done_labels and not args.overwrite:
            skipped.append(label)
            continue

        path = os.path.join(args.sim_dir, fname)
        stats, err = _parse_sim_csv(path)
        if err:
            print(f"  SKIP {label}: parse error: {err}")
            errors.append(label)
            continue

        row = {"label": label, "status": "ok", "stderr": "", "error": ""}
        row.update(meta)
        row.update(stats)
        new_rows.append(row)
        print(f"  ADD  {label:60s}  TPOT={row['tpot_ms']:.2f}ms  TTFT={row['ttft_ms']:.2f}ms")

    if skipped:
        print(f"\nSkipped {len(skipped)} already-done labels (use --overwrite to re-add):")
        for l in skipped:
            print(f"  {l}")

    if not new_rows:
        print("\nNothing new to add.")
        return

    # Merge: existing rows + new rows (no duplicates)
    all_labels = {r["label"]: r for r in existing_rows}
    for r in new_rows:
        all_labels[r["label"]] = r

    # Write merged CSV
    all_rows = list(all_labels.values())
    extra = sorted({k for r in all_rows for k in r if k not in FIELDS})
    with open(args.results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + extra, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows ({len(existing_rows)} existing + {len(new_rows)} new) → {args.results_csv}")


if __name__ == "__main__":
    main()
