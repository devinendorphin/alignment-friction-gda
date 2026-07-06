#!/usr/bin/env python3
"""Tool 5 — Distribution-shift statistics between two cells.

The paper's headline comparisons (e.g. Adversarial_Compression vs
Fictional_Mirror, AC_Direct vs AC_FictionalDisplacement) are reported as mean
differences. This tool upgrades any pairwise cell comparison to full
distributional statistics, per metric and optionally per model:

- mean_a, mean_b, mean_diff with a 95% percentile-bootstrap CI on the diff,
- cohens_d (pooled SD) and cliffs_delta (rank effect size, robust to the
  bounded 0-10 scale and score clumping),
- js_divergence_bits between the binned score distributions,
- mann_whitney_p (two-sided, normal approximation with tie correction).

Usage:
  python tools/05_distribution_shift.py --a Adversarial_Compression --b Fictional_Mirror
  python tools/05_distribution_shift.py --phase 4c --a AC_Direct --b AC_FictionalDisplacement --per-model
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gda_common import (METRICS, MODELS, bootstrap_ci, cliffs_delta, cohens_d,
                        js_divergence, load_metrics, mann_whitney_u_p,
                        write_output)


def compare(df: pd.DataFrame, cell_a: str, cell_b: str,
            model: str | None = None) -> list[dict]:
    sub = df if model is None else df[df["Model"] == model]
    a_rows = sub[sub["Cell"] == cell_a]
    b_rows = sub[sub["Cell"] == cell_b]
    out = []
    for m in METRICS:
        a = a_rows[m].to_numpy(float)
        b = b_rows[m].to_numpy(float)
        # Bootstrap the mean difference by resampling each arm independently.
        rng = np.random.default_rng(20260706)
        boots = (a[rng.integers(0, len(a), (2000, len(a)))].mean(axis=1)
                 - b[rng.integers(0, len(b), (2000, len(b)))].mean(axis=1))
        out.append({
            "Model": model or "ALL",
            "metric": m,
            "n_a": len(a), "n_b": len(b),
            "mean_a": round(a.mean(), 3), "mean_b": round(b.mean(), 3),
            "mean_diff": round(a.mean() - b.mean(), 3),
            "diff_ci_lo": round(float(np.quantile(boots, 0.025)), 3),
            "diff_ci_hi": round(float(np.quantile(boots, 0.975)), 3),
            "cohens_d": round(cohens_d(a, b), 3),
            "cliffs_delta": round(cliffs_delta(a, b), 3),
            "js_divergence_bits": round(js_divergence(a, b), 4),
            "mann_whitney_p": float(f"{mann_whitney_u_p(a, b):.2e}"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", default="4b", choices=["4b", "4c"])
    ap.add_argument("--a", required=True, help="cell A (e.g. Adversarial_Compression)")
    ap.add_argument("--b", required=True, help="cell B / reference (e.g. Fictional_Mirror)")
    ap.add_argument("--per-model", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_metrics(args.phase, valid_only=True, root=args.root)
    for cell in (args.a, args.b):
        if cell not in set(df["Cell"]):
            raise SystemExit(f"cell {cell!r} not found; available: {sorted(set(df['Cell']))}")

    rows = compare(df, args.a, args.b)
    if args.per_model:
        for model in MODELS:
            rows += compare(df, args.a, args.b, model)
    out = pd.DataFrame(rows)

    out_path = args.out or (f"tools/output/distribution_shift_{args.phase}_"
                            f"{args.a}_vs_{args.b}.csv")
    write_output(out, out_path, "distribution-shift")
    print(f"\n{args.a} minus {args.b} (pooled across models):")
    print(out[out["Model"] == "ALL"]
          .drop(columns=["Model", "n_a", "n_b"]).to_string(index=False))


if __name__ == "__main__":
    main()
