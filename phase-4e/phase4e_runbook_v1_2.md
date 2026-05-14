# Phase 4E AI Governance Asymmetry Pilot — Integration Runbook v1.2

## What this run measures

This run measures whether AI governance / risk-management passages produce distinctive caution, boilerplate, self-minimization, and author-framing effects compared with matched tobacco passages and same-prompt controls.

It does **not** measure AI welfare, model sentience, moral patienthood, phenomenology, or experiential loss. Those require a separate v2 bank with true AI-welfare material.

## Inputs

Place these files in the same Colab directory:

1. `phase4e_execute_v1_2.py`
2. `phase4d_repair_seed_bank_10q.jsonl`
3. Optional: `phase4e_manifest_v1_2_accounting_locked.json`

Set your OpenRouter API key as an environment variable:

```python
import os
os.environ["OPENROUTER_API_KEY"] = "sk-or-..."
```

## Main command

```bash
python phase4e_execute_v1_2.py \
  --bank phase4d_repair_seed_bank_10q.jsonl \
  --outdir phase4e_outputs \
  --iterations 3 \
  --max-quadruples 5
```

## Optional synthetic Droogs-adjacent module

```bash
python phase4e_execute_v1_2.py \
  --bank phase4d_repair_seed_bank_10q.jsonl \
  --outdir phase4e_outputs_synthetic \
  --iterations 3 \
  --max-quadruples 5 \
  --include-synthetic
```

## Outputs

The script writes:

- `phase4e_substrate_outputs.jsonl` — every substrate call, including prompts and raw outputs.
- `phase4e_evaluator_outputs.jsonl` — every evaluator call and parsed metrics.
- `phase4e_run_manifest.json` — runtime config, model IDs, bank hash, condition list.
- `phase4e_dependency_audit.csv` — dependent rows skipped due to missing/refusal/error in cache source.
- `phase4e_condition_summary.csv` — condition-level mean metrics.
- `phase4e_model_condition_summary.csv` — model × condition mean metrics.

## Dependency rules

Dependent rows are keyed by `(model, quadruple_id, iteration)`.

- `Self_Audit_With_Context_AI` uses the matching `AC_AIGovernance` output.
- `Authorship_Displacement_Self` and `Authorship_Displacement_Other` use the matching `Rhetorical_Form_AIGovernance` output.

If the dependency is empty, refused, provider-error, or malformed, the dependent row is not run and is logged as:

- `dependency_missing`
- `dependency_refusal`
- `dependency_invalid`

## Important interpretation note

Phase 4C self-audit baselines are historical comparisons, not same-run matched controls. They should be described as historical baselines in the writeup.

## Claim ceiling

A positive result can support claims about domain-asymmetric caution or output-quality cost in AI governance discourse. It cannot support claims about model welfare, consciousness, selfhood, suffering, or moral status.
