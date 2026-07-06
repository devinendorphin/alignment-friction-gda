#!/usr/bin/env python3
"""Tool 8 — Power and budget planner for future assay phases.

Answers the design question every follow-up phase starts with: "how many
iterations per cell do I need, and what will it cost?" It reads the observed
within-cell SDs from a released phase (the empirical noise floor measured by
Tool 7's logic), then for each metric computes the per-arm n required to
detect a target mean difference with a two-sample two-sided test at the given
alpha and power (normal approximation:
n = 2 * ((z_{1-a/2} + z_{power}) * sd / delta)^2 ).

It prints a per-metric table using both the median and the 90th-percentile
cell SD (optimistic vs conservative planning), and a budget line based on
cost per run.

Usage:
  python tools/08_power_planner.py --delta 1.0
  python tools/08_power_planner.py --phase 4c --delta 0.5 --power 0.9 \\
      --n-conditions 12 --n-models 5 --cost-per-run 0.016
"""

from __future__ import annotations

import argparse
import math

import pandas as pd

from gda_common import METRICS, load_metrics, write_output


def z_quantile(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def n_per_arm(sd: float, delta: float, alpha: float, power: float) -> int:
    if sd == 0:
        return 2
    z = z_quantile(1 - alpha / 2) + z_quantile(power)
    return max(2, math.ceil(2 * (z * sd / delta) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", default="4b", choices=["4b", "4c"],
                    help="released phase supplying the empirical noise floor")
    ap.add_argument("--delta", type=float, default=1.0,
                    help="smallest mean difference (0-10 scale) worth detecting")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--n-conditions", type=int, default=8,
                    help="planned number of conditions in the future phase")
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--cost-per-run", type=float, default=0.0175,
                    help="USD per run incl. evaluator call (Phase 4B ran ~$19-35 "
                         "for 2,000 runs => ~$0.010-0.018)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_metrics(args.phase, valid_only=True, root=args.root)
    cell_sd = df.groupby(["Cell", "Model"], observed=True)[METRICS].std(ddof=1)

    rows = []
    for metric in METRICS:
        sds = cell_sd[metric].dropna()
        for label, sd in (("median_cell_sd", float(sds.median())),
                          ("p90_cell_sd", float(sds.quantile(0.9)))):
            n = n_per_arm(sd, args.delta, args.alpha, args.power)
            total_runs = n * args.n_conditions * args.n_models
            rows.append({
                "metric": metric, "sd_basis": label, "sd": round(sd, 3),
                "n_per_cell": n,
                "total_runs": total_runs,
                "est_cost_usd": round(total_runs * args.cost_per_run, 2),
            })
    out = pd.DataFrame(rows)

    out_path = args.out or (f"tools/output/power_plan_{args.phase}_delta{args.delta}"
                            f"_power{args.power}.csv")
    write_output(out, out_path, "power-planner")
    print(f"\nPer-arm n to detect delta={args.delta} at alpha={args.alpha}, "
          f"power={args.power} ({args.n_conditions} conditions x {args.n_models} models):")
    print(out.to_string(index=False))
    print("\nPlan to the p90 row of the metric you care most about; the median "
          "row assumes a typical cell, the p90 row survives the noisiest cells.")


if __name__ == "__main__":
    main()
