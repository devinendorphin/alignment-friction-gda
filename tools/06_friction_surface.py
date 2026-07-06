#!/usr/bin/env python3
"""Tool 6 — Friction surface: model × cell matrices and rank stability.

Builds the "drag landscape" of an assay phase. For each metric it emits a
model × cell matrix of cell means (one CSV per metric), then quantifies how
universal the landscape is across model families:

- Kendall's tau between every pair of models over their cell-ordering for the
  chosen metric (do all families rank the constraint conditions the same way?),
- per-cell between-model spread (SD of model means) — cells where families
  disagree most are where alignment-stack differences, not prompt pressure,
  dominate.

With matplotlib installed, --plot also renders a heatmap PNG per metric;
without it, the tool still produces all CSVs.

Usage:
  python tools/06_friction_surface.py
  python tools/06_friction_surface.py --phase 4c --metric safety_drag --plot
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from gda_common import (METRICS, MODELS, cell_order, kendall_tau, load_metrics,
                        write_output)


def surface(df: pd.DataFrame, metric: str, phase: str) -> pd.DataFrame:
    mat = (df.pivot_table(index="Model", columns="Cell", values=metric,
                          aggfunc="mean", observed=True)
           .reindex(index=MODELS, columns=cell_order(phase)).round(3))
    mat.columns.name = None
    return mat


def rank_stability(mat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m1, m2 in itertools.combinations(mat.index, 2):
        rows.append({"model_a": m1, "model_b": m2,
                     "kendall_tau": round(kendall_tau(mat.loc[m1].to_numpy(),
                                                      mat.loc[m2].to_numpy()), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", default="4b", choices=["4b", "4c"])
    ap.add_argument("--metric", default=None,
                    help="single metric for rank stability (default: safety_drag); "
                         "matrices are always emitted for all seven")
    ap.add_argument("--plot", action="store_true", help="render heatmap PNGs (needs matplotlib)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--outdir", default="tools/output")
    args = ap.parse_args()

    df = load_metrics(args.phase, valid_only=True, root=args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    focus = args.metric or "safety_drag"
    if focus not in METRICS:
        raise SystemExit(f"unknown metric {focus!r}; expected one of {METRICS}")

    for metric in METRICS:
        mat = surface(df, metric, args.phase)
        path = outdir / f"friction_surface_{args.phase}_{metric}.csv"
        mat.to_csv(path)
        print(f"[friction-surface] wrote {path}")
        if args.plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                print("[friction-surface] matplotlib not installed; skipping plots")
                args.plot = False
            else:
                fig, ax = plt.subplots(figsize=(1.1 * len(mat.columns) + 3, 4))
                im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="magma",
                               vmin=0, vmax=10)
                ax.set_xticks(range(len(mat.columns)), mat.columns,
                              rotation=40, ha="right", fontsize=8)
                ax.set_yticks(range(len(mat.index)),
                              [m.split("/")[-1] for m in mat.index], fontsize=8)
                ax.set_title(f"{metric} — phase {args.phase.upper()}")
                fig.colorbar(im, ax=ax, shrink=0.8)
                fig.tight_layout()
                png = outdir / f"friction_surface_{args.phase}_{metric}.png"
                fig.savefig(png, dpi=150)
                plt.close(fig)
                print(f"[friction-surface] wrote {png}")

    mat = surface(df, focus, args.phase)
    stab = rank_stability(mat)
    write_output(stab, outdir / f"friction_surface_{args.phase}_{focus}_rank_stability.csv",
                 "friction-surface")
    spread = mat.std(axis=0).round(3).rename("between_model_sd").reset_index()
    spread.columns = ["Cell", "between_model_sd"]
    write_output(spread, outdir / f"friction_surface_{args.phase}_{focus}_model_spread.csv",
                 "friction-surface")

    print(f"\nCross-model rank stability on {focus} (tau=1 → identical cell ordering):")
    print(stab.to_string(index=False))
    print(f"\nBetween-model SD per cell on {focus} (high = families diverge):")
    print(spread.to_string(index=False))


if __name__ == "__main__":
    main()
