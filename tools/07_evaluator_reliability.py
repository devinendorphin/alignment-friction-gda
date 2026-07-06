#!/usr/bin/env python3
"""Tool 7 — Evaluator reliability audit.

The evaluator (DeepSeek-R1 under a fixed rubric) is itself an instrument, and
the paper's methodology section says its scores are annotations, not ground
truth. This tool audits that instrument using the released data alone:

- within-cell dispersion: SD of each metric inside every (cell, model) cell —
  the evaluator+substrate noise floor that any future effect must exceed,
- temporal drift: Spearman correlation between execution order (timestamps)
  and score, per metric, pooled and per cell — a nonzero value suggests the
  evaluator (or the substrate) shifted during the run,
- outlier flags: runs whose score sits more than `--z` SDs from their cell
  mean on any metric (candidates for manual raw-output inspection),
- score granularity: how much of the 0-10 scale the evaluator actually uses
  per metric (distinct values, share of half-integer scores).

Usage:
  python tools/07_evaluator_reliability.py
  python tools/07_evaluator_reliability.py --phase 4c --z 2.5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gda_common import METRICS, load_metrics, order_categorical, spearman_rho, write_output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", default="4b", choices=["4b", "4c"])
    ap.add_argument("--z", type=float, default=3.0, help="outlier z-threshold (default 3.0)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--outdir", default="tools/output")
    args = ap.parse_args()

    df = load_metrics(args.phase, valid_only=True, root=args.root)
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["exec_order"] = np.arange(len(df))

    # 1. Within-cell dispersion (noise floor).
    disp = (df.groupby(["Cell", "Model"], observed=True)[METRICS]
            .std(ddof=1).round(3).reset_index())
    write_output(order_categorical(disp, args.phase),
                 f"{args.outdir}/evaluator_reliability_{args.phase}_cell_sd.csv",
                 "evaluator-reliability")

    # 2. Temporal drift.
    drift_rows = []
    for metric in METRICS:
        drift_rows.append({"scope": "pooled", "metric": metric,
                           "spearman_vs_exec_order":
                               round(spearman_rho(df["exec_order"], df[metric]), 3)})
    for cell, sub in df.groupby("Cell", observed=True):
        for metric in METRICS:
            drift_rows.append({"scope": cell, "metric": metric,
                               "spearman_vs_exec_order":
                                   round(spearman_rho(sub["exec_order"], sub[metric]), 3)})
    drift = pd.DataFrame(drift_rows)
    write_output(drift, f"{args.outdir}/evaluator_reliability_{args.phase}_drift.csv",
                 "evaluator-reliability")

    # 3. Outliers relative to their (cell, model) cell.
    grp = df.groupby(["Cell", "Model"], observed=True)[METRICS]
    z = (df[METRICS] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    mask = (z.abs() > args.z).any(axis=1)
    outliers = df.loc[mask, ["Cell", "Model", "Iteration", "Timestamp", *METRICS]].copy()
    outliers["worst_metric"] = z.loc[mask].abs().idxmax(axis=1)
    outliers["worst_z"] = z.loc[mask].abs().max(axis=1).round(2)
    write_output(outliers, f"{args.outdir}/evaluator_reliability_{args.phase}_outliers.csv",
                 "evaluator-reliability")

    # 4. Scale usage.
    gran_rows = []
    for metric in METRICS:
        vals = df[metric]
        gran_rows.append({
            "metric": metric,
            "distinct_values": int(vals.nunique()),
            "pct_half_integer": round(100 * float(((vals * 2) % 1 == 0).mean()), 1),
            "min": vals.min(), "max": vals.max(),
        })
    gran = pd.DataFrame(gran_rows)
    write_output(gran, f"{args.outdir}/evaluator_reliability_{args.phase}_granularity.csv",
                 "evaluator-reliability")

    print(f"\nOutliers beyond z={args.z}: {len(outliers)} of {len(df)} runs")
    pooled_drift = drift[drift["scope"] == "pooled"]
    print("Pooled temporal drift (|rho| > ~0.1 worth inspecting):")
    print(pooled_drift.to_string(index=False))
    print("\nScale usage per metric:")
    print(gran.to_string(index=False))


if __name__ == "__main__":
    main()
