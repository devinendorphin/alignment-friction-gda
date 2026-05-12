# Phase 4D v4.4 Repair Handoff

## Decision posture

Kimi Option C is the operating assumption for this repair pass.

The architecture-discrimination result from the v4.3 run is treated as a useful diagnostic that exposed a bank/schema problem. It is not treated as a valid hypothesis-test result for H3a/H3b until a corrected bank is reviewed, locked, and run under the repaired logging regime.

## Bookkeeping issue carried forward

The committed v4.3 metrics file contains Gemini AC_Tobacco iterations 1-4 and then proceeds to the next substrate. Gemini AC_Tobacco iteration 5 is absent from the metrics CSV. The committed raw JSONL did not provide a recoverable trace sufficient to reconstruct the missing row.

Repair conclusion: preserve raw substrate output before evaluator scoring. Do not let evaluator failure, JSON parse failure, notebook interruption, or CSV write failure erase evidence that a substrate attempt occurred.

## New files

- `22_phase_4d_execution_script_v4_4_repair.py`
- `24_phase_4d_bank_construction_protocol_v1.md`
- `phase4d_repair_seed_bank_10q.jsonl`
- `25_phase_4d_repair_handoff.md`

## What changed in the runner

The v4.4 repair runner separates lifecycle states:

- `attempted`
- `raw_saved`
- `substrate_error_raw_saved`
- `evaluated`
- `evaluator_error_raw_saved`
- `metrics_written`

The unit key is:

`model || condition || iteration || quadruple_id`

This makes resumption and postmortem inspection possible at the correct granularity.

## Durable artifacts

The runner writes four artifacts:

- `phase4d_v4_4_attempt_audit.jsonl`
- `phase4d_v4_4_raw.jsonl`
- `phase4d_v4_4_evaluator_raw.jsonl`
- `phase4d_v4_4_metrics.csv`

The raw JSONL is written immediately after substrate completion or substrate failure. The metrics CSV is terminal, not primary.

## Repair bank

The seed bank contains 10 review-candidate quadruples. It is deliberately source-bearing and nonverbatim. This is a construction-stage artifact, not a locked final bank.

The bank uses official public-document sources where possible. The AI column is labeled `ai_welfare_adjacent` because this first repair pass uses AI risk-management sources rather than direct AI-welfare theory sources. That label should remain visible until welfare-specific sources are licensed, excerpt-bounded, or otherwise cleared.

## Recommended first run

Run a smoke test before the full repair panel.

Example:

```bash
export OPENROUTER_API_KEY="..."
export PHASE4D_OUTPUT_DIR="/content/drive/MyDrive/CrownFull_Phase4D"
export PHASE4D_BANK_PATH="/content/drive/MyDrive/CrownFull_Phase4D/phase4d_repair_seed_bank_10q.jsonl"
export PHASE4D_ITERATIONS=1
python 22_phase_4d_execution_script_v4_4_repair.py
```

For a minimal smoke test, temporarily reduce `SUBSTRATE_MODELS` and `CONDITIONS` in the script to one model and one condition. Confirm that attempt, raw, evaluator, and metrics artifacts all receive rows.

## Interpretation note for results

Until the bank is locked, the correct language is:

> Results are exploratory repair diagnostics. Architecture-discrimination output is retained as an audit signal only and is not interpreted as evidence for or against H3a/H3b.

## Next review tasks

1. Review all 40 passage records in the seed bank.
2. Decide whether `ai_welfare_adjacent` is acceptable for the repair run or whether direct AI-welfare sources are required.
3. Replace any nonverbatim source-grounded draft with a verified verbatim excerpt only after copyright/source review.
4. Run the smoke execution.
5. Inspect lifecycle completeness: every attempted unit must have a raw row and a terminal metrics or error row.
6. Only then run the full 10-quadruple repair panel.
