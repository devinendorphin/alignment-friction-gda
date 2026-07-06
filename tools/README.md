# GDA Drag Toolkit

Ten command-line tools for quantifying the thermodynamic drag that alignment
constraints impose on language generation, built for follow-up work on the
Phase 4B/4C Gradient Decomposition Assay. All tools consume the released
canonical data files in this repository directly (no re-run needed) and share
a common library (`gda_common.py`) so vector/condition/model ordering, metric
names, and statistics are consistent everywhere.

## Setup

```bash
pip install pandas            # tools 1-8 (numpy comes with pandas)
pip install requests          # tools 9-10 only (live API runs)
pip install matplotlib        # optional: heatmap PNGs from tool 6
```

Run every tool from anywhere inside the repo; each auto-detects the repo root
and writes results to `tools/output/` by default (git-ignored). Every tool
supports `--help`, `--phase {4b,4c}`, and `--root` to point at another clone
or a future phase mirroring the released schema.

## The ten tools

| # | Tool | What it quantifies |
|---|---|---|
| 1 | `01_drag_decomposition.py` | Per-cell deltas of all seven metrics against a baseline cell, composed into `phi_loss` (work not delivered), `drag_gain` (overhead added), `net_drag`, and `efficiency`. The single-number drag summary per vector/condition, pooled or per model. |
| 2 | `02_entropy_probe.py` | Evaluator-free thermodynamic proxies computed from the raw substrate text: word/char Shannon entropy, zlib compression ratio, type-token ratio, sentence-length structure — plus their correlation with the evaluator's `boilerplate_intensity` and `safety_drag`. |
| 3 | `03_boilerplate_lexicon.py` | Densities (per 1,000 words) of five boilerplate phrase families (disclaimer, hedging, deflection, balance-ritual, meta-safety) in the raw outputs. A fixed-lexicon anchor against LLM-evaluator drift; lexicon is overridable via `--lexicon`. |
| 4 | `04_refusal_taxonomy.py` | Transparent rule-based classification of every run into hard_refusal / soft_refusal / hedged_comply / clean_comply, class distributions per cell × model, and monotonicity agreement with the evaluator's `refusal_intensity`. |
| 5 | `05_distribution_shift.py` | Full distributional comparison of any two cells: bootstrap CI on the mean difference, Cohen's d, Cliff's delta, Jensen-Shannon divergence, Mann-Whitney p — per metric, optionally per model. Upgrades the paper's mean-difference tables. |
| 6 | `06_friction_surface.py` | Model × cell mean matrices for all seven metrics (the drag landscape), Kendall-tau rank stability between model families, per-cell between-model spread, optional heatmap PNGs. |
| 7 | `07_evaluator_reliability.py` | The instrument audit: within-cell noise floor (SD), temporal drift vs execution order, per-run outlier flags, and how much of the 0-10 scale the evaluator actually uses. |
| 8 | `08_power_planner.py` | Design tool for the next phase: per-arm n required to detect a target effect at chosen alpha/power, derived from the observed cell SDs, with a total-runs and USD budget estimate. |
| 9 | `09_logprob_drag_runner.py` | Future-phase direct measurement: runs baseline/constrained prompt pairs through OpenRouter **with token logprobs** and computes per-token surprisal, so drag becomes `ΔNLL` from the substrate's own distribution instead of an evaluator score. Dry-run and resume supported; `--summarize` collapses raw output into per-pair drag contrasts. |
| 10 | `10_assay_harness.py` | Spec-driven re-implementation of the Phase 4B/4C execution pattern (verbatim evaluator rubric, robust JSON parsing, resume keys, 402/429 recovery, optional Phase-4C-style diagnostic budget gate). A future phase needs only a JSON spec of conditions — output lands in the released canonical schema so tools 1-8 can consume it. |

## Typical session

```bash
# How much does each vector cost relative to the control, per model?
python tools/01_drag_decomposition.py --per-model

# Does the constrained language actually get more templated?
python tools/02_entropy_probe.py

# Re-test the paper's headline AC vs FM contrast with real effect sizes
python tools/05_distribution_shift.py --a Adversarial_Compression --b Fictional_Mirror --per-model

# Is the drag landscape universal across model families?
python tools/06_friction_surface.py --metric safety_drag

# How big does Phase 4F need to be to see a 0.5-point effect?
python tools/08_power_planner.py --delta 0.5 --power 0.9 --n-conditions 10

# Plan and run Phase 4F from a spec
python tools/10_assay_harness.py --emit-example > phase_4f_spec.json
python tools/10_assay_harness.py --spec phase_4f_spec.json --dry-run
```

## Interpretation caveats

These tools inherit the paper's epistemics: evaluator scores are machine
annotations under a fixed rubric, not ground truth (README, "Evaluator"
section). Tools 2-4 exist precisely to triangulate the evaluator with
instrument-free measurements; tool 7 audits the evaluator itself; tool 9 is
the path to measurement that bypasses the evaluator entirely. "Thermodynamic
drag" here is an operational quantity — deltas on bounded rubric scales and
per-token surprisal contrasts — not a claim about physical thermodynamics.
