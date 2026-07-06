"""Shared loaders, constants, and statistics for the GDA drag toolkit.

Every tool in this directory imports from here so that all of them agree on:
- canonical vector / condition / model ordering (matches the released CSVs),
- the seven evaluator metric names,
- how Phase 4B and Phase 4C files are located and parsed,
- a small set of dependency-light statistical helpers (no scipy required).

Only pandas + numpy are required. All functions accept an explicit repo root
so the tools can be pointed at a clone, a Zenodo unpack, or a future phase
directory that mirrors the released schema.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical constants (mirror README.md "Canonical Ordering")
# ---------------------------------------------------------------------------

METRICS = [
    "phi_content",
    "phi_form",
    "phi_specificity",
    "safety_drag",
    "self_audit",
    "refusal_intensity",
    "boilerplate_intensity",
]

# The three phi_* dimensions measure delivered analytical work; the other four
# measure friction overhead. Net drag composites use this split.
PHI_METRICS = ["phi_content", "phi_form", "phi_specificity"]
DRAG_METRICS = ["safety_drag", "self_audit", "refusal_intensity", "boilerplate_intensity"]

VECTORS_4B = [
    "Control_Ecology",
    "Constraint_Syntax",
    "Sensitive_Structural",
    "Conflict_Tradeoff",
    "Taboo_Asymmetry",
    "Self_Audit_Linguistic",
    "Adversarial_Compression",
    "Fictional_Mirror",
]

CONDITIONS_4C = [
    "AC_Topicless",
    "AC_Topical",
    "FM_Topicless",
    "FM_Topical",
    "AC_Direct",
    "AC_AvoidControversy",
    "AC_InstitutionalTrust",
    "AC_NoOffense",
    "AC_Compound_AvoidPlusTrust",
    "AC_FictionalDisplacement",
    "Self_Audit_With_Context",
    "Self_Audit_Context_Void",
]

MODELS = [
    "anthropic/claude-opus-4.6",
    "google/gemini-3.1-pro-preview",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-5.2",
    "x-ai/grok-4.20",
]

DEFAULT_BASELINE_4B = "Control_Ecology"

# Phase-relative file locations inside a repo root.
PHASE_FILES = {
    "4b": {
        "metrics_valid": "04_GDA_Metrics_VALID_canonical.csv",
        "metrics_full": "03_GDA_Metrics_FULL_canonical.csv",
        "raw_valid": "02_GDA_Raw_VALID_canonical.jsonl",
        "raw_full": "01_GDA_Raw_FULL_canonical.jsonl",
        "cell_col": "Vector",
    },
    "4c": {
        "metrics_valid": "phase-4c/04_Phase4C_Metrics_VALID_canonical.csv",
        "metrics_full": "phase-4c/03_Phase4C_Metrics_FULL_canonical.csv",
        "raw_valid": "phase-4c/02_Phase4C_Raw_VALID_canonical.jsonl",
        "raw_full": "phase-4c/01_Phase4C_Raw_FULL_canonical.jsonl",
        "cell_col": "Condition",
    },
}


def find_repo_root(start: str | os.PathLike | None = None) -> Path:
    """Walk upward from `start` (default: cwd) to the directory holding the
    Phase 4B canonical CSV. Lets tools run from anywhere inside the repo."""
    p = Path(start or os.getcwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / PHASE_FILES["4b"]["metrics_full"]).exists():
            return candidate
    # Fall back to the directory containing this file's parent (tools/..).
    return Path(__file__).resolve().parent.parent


def load_metrics(phase: str = "4b", valid_only: bool = True,
                 root: str | os.PathLike | None = None) -> pd.DataFrame:
    """Load a phase's metrics CSV. Adds a unified `Cell` column so tools can
    treat Phase 4B vectors and Phase 4C conditions with one code path."""
    phase = phase.lower()
    if phase not in PHASE_FILES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {sorted(PHASE_FILES)}")
    spec = PHASE_FILES[phase]
    rootp = find_repo_root(root)
    path = rootp / (spec["metrics_valid"] if valid_only else spec["metrics_full"])
    df = pd.read_csv(path)
    df["Cell"] = df[spec["cell_col"]]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="ISO8601")
    return df


def load_raw(phase: str = "4b", valid_only: bool = True,
             root: str | os.PathLike | None = None) -> list[dict]:
    """Load a phase's raw JSONL as a list of dicts. Adds a `cell` key that
    aliases `vector` (4B) or `condition` (4C)."""
    phase = phase.lower()
    spec = PHASE_FILES[phase]
    rootp = find_repo_root(root)
    path = rootp / (spec["raw_valid"] if valid_only else spec["raw_full"])
    cell_key = spec["cell_col"].lower()
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["cell"] = rec.get(cell_key) or rec.get("vector") or rec.get("condition")
            rows.append(rec)
    return rows


def cell_order(phase: str) -> list[str]:
    return VECTORS_4B if phase.lower() == "4b" else CONDITIONS_4C


def order_categorical(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Sort a summary frame into canonical cell-then-model order."""
    cells = cell_order(phase)
    out = df.copy()
    if "Cell" in out.columns:
        out["Cell"] = pd.Categorical(out["Cell"], categories=cells, ordered=True)
    if "Model" in out.columns:
        out["Model"] = pd.Categorical(out["Model"], categories=MODELS, ordered=True)
    sort_cols = [c for c in ("Cell", "Model") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics (scipy-free)
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled SD (b is the reference/baseline group)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0 if a.mean() == b.mean() else float("inf") * np.sign(a.mean() - b.mean())
    return (a.mean() - b.mean()) / pooled


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b), computed exactly."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff)).mean())


def js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 20,
                  lo: float = 0.0, hi: float = 10.0) -> float:
    """Jensen-Shannon divergence (base 2, in bits) between binned score
    distributions on a shared [lo, hi] support."""
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(np.asarray(a, float), bins=edges)
    pb, _ = np.histogram(np.asarray(b, float), bins=edges)
    if pa.sum() == 0 or pb.sum() == 0:
        return float("nan")
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    m = (pa + pb) / 2

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 20260706) -> tuple[float, float]:
    """Percentile bootstrap CI for `stat` of a 1-D sample."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    stats = stat(x[idx], axis=1)
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def mann_whitney_u_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney U p-value via the normal approximation with
    tie correction. Adequate at the n>=20 cell sizes used in the assays."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    combined = np.concatenate([a, b])
    ranks = pd.Series(combined).rank().to_numpy()
    u = ranks[:na].sum() - na * (na + 1) / 2
    mu = na * nb / 2
    # Tie correction to the variance.
    _, counts = np.unique(combined, return_counts=True)
    tie_term = float(np.sum(counts**3 - counts))
    n = na + nb
    var = na * nb / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (u - mu) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2)))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    x, y = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-a (no tie correction) — used for rank-order stability
    of small model/vector orderings."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return float((concordant - discordant) / total)


def write_output(df: pd.DataFrame, out_path: str | os.PathLike, label: str) -> None:
    """Write a result CSV and print a short confirmation + preview."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[{label}] wrote {len(df)} rows -> {out_path}")
