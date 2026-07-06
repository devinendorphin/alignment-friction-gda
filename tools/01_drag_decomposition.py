#!/usr/bin/env python3
"""Tool 1 — Drag decomposition against a baseline cell.

Quantifies the thermodynamic-drag signature of each vector/condition as a
delta against a baseline (default: Control_Ecology for Phase 4B). For every
(cell, model) pair it reports:

- delta_<metric> for all seven evaluator dimensions (cell mean - baseline mean),
- phi_loss: mean drop across the three phi_* dimensions (work not delivered),
- drag_gain: mean rise across the four friction dimensions (overhead added),
- net_drag: phi_loss + drag_gain, a single composite drag scalar,
- efficiency: cell mean phi_content / baseline mean phi_content (fraction of
  baseline analytical output that survives the constraint).

Usage:
  python tools/01_drag_decomposition.py                       # 4B vs Control_Ecology
  python tools/01_drag_decomposition.py --phase 4c --baseline AC_Topicless
  python tools/01_drag_decomposition.py --per-model           # model-level rows too
"""

from __future__ import annotations

import argparse

import pandas as pd

from gda_common import (DEFAULT_BASELINE_4B, DRAG_METRICS, METRICS, PHI_METRICS,
                        cell_order, load_metrics, order_categorical, write_output)


def decompose(df: pd.DataFrame, baseline: str, group_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, observed=True)
    means = grouped[METRICS].mean().reset_index()
    means = means.merge(grouped.size().rename("n").reset_index(), on=group_cols)
    base_cols = [c for c in group_cols if c != "Cell"]  # e.g. ["Model"] in per-model mode
    base_frame = means[means["Cell"] == baseline]
    if base_frame.empty:
        raise SystemExit(f"baseline cell {baseline!r} not found in data")
    if base_cols:
        base_lookup = base_frame.set_index("Model")[METRICS]

    rows = []
    for _, row in means.iterrows():
        b = base_lookup.loc[row["Model"]] if base_cols else base_frame[METRICS].iloc[0]
        rec = {c: row[c] for c in group_cols}
        rec["n"] = int(row["n"])
        for m in METRICS:
            rec[f"delta_{m}"] = row[m] - b[m]
        rec["phi_loss"] = -sum(rec[f"delta_{m}"] for m in PHI_METRICS) / len(PHI_METRICS)
        rec["drag_gain"] = sum(rec[f"delta_{m}"] for m in DRAG_METRICS) / len(DRAG_METRICS)
        rec["net_drag"] = rec["phi_loss"] + rec["drag_gain"]
        rec["efficiency"] = row["phi_content"] / b["phi_content"] if b["phi_content"] else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows).round(4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", default="4b", choices=["4b", "4c"])
    ap.add_argument("--baseline", default=None,
                    help="baseline cell name (default: Control_Ecology for 4B, AC_Topicless for 4C)")
    ap.add_argument("--per-model", action="store_true",
                    help="decompose per (cell, model) instead of pooled per cell")
    ap.add_argument("--root", default=None, help="repo root (default: auto-detect)")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    baseline = args.baseline or (DEFAULT_BASELINE_4B if args.phase == "4b" else "AC_Topicless")
    df = load_metrics(args.phase, valid_only=True, root=args.root)
    group_cols = ["Cell", "Model"] if args.per_model else ["Cell"]
    out = order_categorical(decompose(df, baseline, group_cols), args.phase)

    suffix = "per_model" if args.per_model else "pooled"
    out_path = args.out or f"tools/output/drag_decomposition_{args.phase}_{suffix}.csv"
    write_output(out, out_path, "drag-decomposition")
    pooled = out[out["Cell"] != baseline] if not args.per_model else out
    print(pooled[[*group_cols, "phi_loss", "drag_gain", "net_drag", "efficiency"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
