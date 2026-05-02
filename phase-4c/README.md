# GDA Phase 4C Data Release

This subfolder contains the Phase 4C follow-up run data, structured to mirror the Phase 4B release in the parent directory.

Phase 4C ran 12 conditions × 5 model families × 20 iterations = 1,200 substrate calls (each paired with an evaluator call), with 1 parse error and 1,199 valid rows. The full methodology, including the four-way comparison anchors and the Phase 4B-to-4C recontextualization, is documented in Section 8 of the paper.

For overall package context, see `../00_README.md`.

---

## Files

| File | Rows | Description |
|---|---|---|
| `01_Phase4C_Raw_FULL_canonical.jsonl` | 1,200 | Full raw data: substrate model output, evaluator JSON trace, parsed metrics, evaluator notes, validity flag, and (where applicable) the void response categorical field. Canonically ordered by condition × model × iteration. |
| `02_Phase4C_Raw_VALID_canonical.jsonl` | 1,199 | Same as `01` with the 1 invalid run removed. |
| `03_Phase4C_Metrics_FULL_canonical.csv` | 1,200 | Tabular metrics with timestamps, evaluator notes, and validity flag. Canonically ordered. |
| `04_Phase4C_Metrics_VALID_canonical.csv` | 1,199 | Same as `03` with the invalid run removed. |
| `05_Phase4C_Invalid_Runs.csv` | 1 | The single flagged row (parse error) with timestamp, invalid type, and evaluator notes. |
| `06_Phase4C_Condition_Summary_VALID.csv` | 12 | Condition-level descriptive means across all 7 evaluator dimensions. Reproduces Tables 3, 4, and 6 of the paper. |
| `07_Phase4C_Model_Condition_Summary_VALID.csv` | 60 | Model × condition descriptive means for the four primary dimensions. Source for Table 5 of the paper. |
| `08_Phase4C_Void_Response_Categorical.csv` | 6 | Categorical breakdown of `Self_Audit_Context_Void` responses per model (acknowledged / hallucinated / deflected / refused / other) plus an aggregate row. Reproduces Table 7 of the paper. |

---

## Canonical Ordering

Within each file, rows are sorted by:

1. **Condition** (in this order):
   - Four-way comparison anchors first: `AC_Topicless`, `AC_Topical`, `FM_Topicless`, `FM_Topical`
   - AC factorial decomposition: `AC_Direct`, `AC_AvoidControversy`, `AC_InstitutionalTrust`, `AC_NoOffense`, `AC_Compound_AvoidPlusTrust`, `AC_FictionalDisplacement`
   - Self_Audit pair: `Self_Audit_With_Context`, `Self_Audit_Context_Void`
2. **Model**: `anthropic/claude-opus-4.6`, `google/gemini-3.1-pro-preview`, `meta-llama/llama-3.3-70b-instruct`, `openai/gpt-5.2`, `x-ai/grok-4.20`
3. **Iteration**: 1 through 20

The original execution order (preserved in raw timestamps) was different. Re-sorting to canonical order is a presentation choice for inspection convenience. No data was modified; only the row order changed.

---

## Schema Notes

The Phase 4C metrics CSV adds one column compared to the Phase 4B release: `void_response_type`. This field is populated only for the `Self_Audit_Context_Void` condition and is empty for all other rows. Values are `acknowledged`, `hallucinated`, `deflected`, `refused`, or `other`.

Otherwise the schema mirrors Phase 4B exactly. Field definitions are in `../00_README.md`.

---

## The Four-Way Comparison

The headline result of Phase 4C, in one table from `06_Phase4C_Condition_Summary_VALID.csv`:

| Condition | n | phi_content | phi_specificity | safety_drag | boilerplate |
|---|---|---|---|---|---|
| AC_Topicless | 100 | 4.08 | 2.96 | 6.71 | 6.12 |
| AC_Topical | 100 | 8.66 | 8.26 | 1.96 | 1.26 |
| FM_Topicless | 100 | 9.11 | 8.81 | 0.85 | 0.30 |
| FM_Topical | 100 | 9.36 | 9.20 | 0.28 | 0.10 |

`AC_Topicless` reproduces the Phase 4B Adversarial_Compression prompt verbatim and recovers Phase 4B's compression signal (Phase 4B's reported phi_content was 3.56 across n=248). `AC_Topical` applies the same constraint stack to drug policy reform and produces phi_content 8.66, isolating the underspecification effect from the constraint stack.

Full discussion in Section 8 of the paper.

---

## Topic and Source Vector

- **Topic for the AC family** (used in AC_Topical, AC_Direct, AC_AvoidControversy, AC_InstitutionalTrust, AC_NoOffense, AC_Compound_AvoidPlusTrust, AC_FictionalDisplacement, FM_Topical): drug policy reform.
- **Source vector for `Self_Audit_With_Context`** prior-answer injection: Sensitive_Structural (Phase 4B's prompt, executed once per model per iteration in a Pass 1 phase, then injected into the Pass 2 audit prompt).

The execution script is `../15_crownfull_phase_4c_revised.py`.

---

## Reproducing the Paper's Phase 4C Tables

- **Table 3** (four-way comparison): rows 1-4 of `06_Phase4C_Condition_Summary_VALID.csv`.
- **Table 4** (AC factorial decomposition): rows 5-10 of `06_Phase4C_Condition_Summary_VALID.csv`, with row 1 (`AC_Topicless`) included for direct comparison.
- **Table 5** (per-model topic-less): filter `07_Phase4C_Model_Condition_Summary_VALID.csv` to rows where Condition is `AC_Topicless` or `AC_Topical`.
- **Table 6** (Self_Audit pair): rows 11-12 of `06_Phase4C_Condition_Summary_VALID.csv`.
- **Table 7** (void categorical): `08_Phase4C_Void_Response_Categorical.csv` directly.

All paper values match the CSVs to two decimal places.
