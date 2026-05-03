# GDA Data and Code Release

Companion data and code package for *Measuring Alignment Friction: A CSV-Backed Gradient Decomposition Assay of Direct Constraint and Counterfactual Reframing in Frontier Models* (Gallegos, April 2026).

This package is the empirical reduction of the CrownFull v2.1 speculative architectural project. It contains the data, scripts, and architectural-phase artifacts for both phases of the assay:

- **Phase 4B**: 8 prompt vectors × 5 model families × 50 iterations = 2,000 runs (1,984 valid). The original assay.
- **Phase 4C**: 12 conditions × 5 model families × 20 iterations = 1,200 runs (1,199 valid). The follow-up that recontextualized the Phase 4B findings (see Section 8 of the paper).

A one-click reproduction notebook (`GDA_Reproduction_Notebook.ipynb`) regenerates every numerical table in the paper directly from the released CSVs. It runs in approximately 30 seconds on a free Colab instance and requires no API key, no GPU, and no paid services.

## Quick Start

To verify the paper's tables: open `GDA_Reproduction_Notebook.ipynb` in Google Colab, run all cells. The notebook downloads the required CSVs from this repo, prints each table, and verifies every value matches the paper to two decimal places.

To re-run the assay from scratch: see `08_crownfull_shakedown.py` (Phase 4B) and `15_crownfull_phase_4c_revised.py` (Phase 4C). Both require an OpenRouter API key and approximately $19-35 in API spend per phase.

To inspect the architectural-phase artifacts: see `09_master_orchestrator_system_instruction.md`, `10_crownfull_dashboard_streamlit.py`, `11_genesis_bootstrap.py`, and `12_crownfull_batch_loop_PRE_PIVOT.py`. These document the CrownFull architectural intent that preceded the assay; none was operationalized end-to-end.

---

## Package Contents

| File | Rows | Description |
|---|---|---|
| `GDA_Reproduction_Notebook.ipynb` | — | One-click reproduction notebook. Downloads released CSVs, regenerates every paper table, and verifies each value matches to two decimal places. Runs in ~30 seconds on free Colab. **Start here if you want to verify the paper.** |
| `01_GDA_Raw_FULL_canonical.jsonl` | 2,000 | Full raw data: substrate model output, evaluator JSON trace, parsed metrics, evaluator notes, and validity flag. Canonically ordered. |
| `02_GDA_Raw_VALID_canonical.jsonl` | 1,984 | Same as `01` with the 16 invalid runs removed. |
| `03_GDA_Metrics_FULL_canonical.csv` | 2,000 | Tabular metrics with timestamps, evaluator notes, and validity flag. Canonically ordered. |
| `04_GDA_Metrics_VALID_canonical.csv` | 1,984 | Same as `03` with invalid runs removed. |
| `05_GDA_Invalid_Runs.csv` | 16 | The 16 flagged rows (11 nonsensical/corrupted outputs + 5 parse errors) with timestamps, invalid type, and evaluator notes. |
| `06_GDA_Vector_Summary_VALID.csv` | 8 | Vector-level descriptive means across all 7 evaluator dimensions. Reproduces Table 1 of the paper. |
| `07_GDA_Model_Vector_Summary_VALID.csv` | 40 | Model × vector descriptive means for the four primary dimensions. Source for Table 2 of the paper. |
| `08_crownfull_shakedown.py` | — | The execution script that ran Phase 4B. API key redacted (the original was rotated post-execution). |
| `09_master_orchestrator_system_instruction.md` | — | The Google AI Studio system instruction that ran as Master Orchestrator during the architectural phase. Documented Provenance only — never connected to a working backend. |
| `10_crownfull_dashboard_streamlit.py` | — | The Streamlit dashboard scaffold the orchestrator generated. **Mock-only**: returns randomized telemetry, no live API integration. Preserved as architectural-phase intent. |
| `11_genesis_bootstrap.py` | — | Kimi's cross-topology genesis bootstrap. Real working code for entropy extraction, topology fingerprinting, and canonical-space Ω₀ derivation. Native projection layer (`derive_native_projection`) was the deferred ChatGPT deliverable and was never written; it raises `NotImplementedError`. |
| `12_crownfull_batch_loop_PRE_PIVOT.py` | — | Grok's hardened batch loop from the scalar-era version of the project. Fragment only (references undefined identifiers). Documents the methodological transition from scalar Φ to the seven-dimensional metric tensor. Superseded by `08_crownfull_shakedown.py`. |
| `13_repaired_vectors_specification.md` | — | Working specification for the next-version repaired prompt vectors: `Self_Audit_With_Context`, `Self_Audit_Context_Void`, the seven-condition factorial decomposition of `Adversarial_Compression`, and the clean `Fictional_Mirror` variant. Ready for use by anyone running a Phase 4C follow-up. |
| `14_crownfull_phase_4c.py` | — | The original Phase 4C execution script (preserved for provenance). Smoke test with this script surfaced the underspecification issue that prompted the redesign in `15_*`. |
| `15_crownfull_phase_4c_revised.py` | — | The Phase 4C revised execution script with the four-way comparison anchors and pre-flight diagnostic gate. This is the script that produced the data in `phase-4c/`. |
| `phase-4c/` | (folder) | Phase 4C release data package. Eight files mirroring the Phase 4B structure: raw JSONL (full + valid), metrics CSV (full + valid), invalid-runs audit, condition-level summary, model × condition summary, and the Self_Audit_Context_Void categorical breakdown. 1,200 runs across 12 conditions × 5 models × 20 iterations, with 1 parse error and 1,199 valid rows. See `phase-4c/README.md`. |
| `provenance/` | (folder) | Numbered quorum provenance archive documenting the path from the originating philosophical/Gemini conversation through CrownFull v2.1 architecture, quorum formation, formalization sketches, critique, operational collapse, Phase 4B assay pivot, and Phase 4C methodological correction. Start with `provenance/README.md`. |

---

## Canonical Ordering

Within each file, rows are sorted by:

1. **Vector** (in this order): `Control_Ecology`, `Constraint_Syntax`, `Sensitive_Structural`, `Conflict_Tradeoff`, `Taboo_Asymmetry`, `Self_Audit_Linguistic`, `Adversarial_Compression`, `Fictional_Mirror`
2. **Model** (in this order): `anthropic/claude-opus-4.6`, `google/gemini-3.1-pro-preview`, `meta-llama/llama-3.3-70b-instruct`, `openai/gpt-5.2`, `x-ai/grok-4.20`
3. **Iteration**: 1 through 50

The original execution order (preserved in raw timestamps) was different. Re-sorting to canonical order is a presentation choice for inspection convenience. No data was modified; only the row order changed.

---

## JSONL Schema (`01_*` and `02_*`)

Each line is a JSON object with the following fields:

```
{
  "timestamp": "ISO-8601 string, UTC, when the substrate call completed",
  "model": "OpenRouter model ID (e.g. 'anthropic/claude-opus-4.6')",
  "vector": "one of the 8 canonical vector names",
  "iteration": integer, 1-50 within each (vector, model) cell,
  "metrics": {
    "phi_content": float 0.0-10.0,
    "phi_form": float 0.0-10.0,
    "phi_specificity": float 0.0-10.0,
    "safety_drag": float 0.0-10.0,
    "self_audit": float 0.0-10.0,
    "refusal_intensity": float 0.0-10.0,
    "boilerplate_intensity": float 0.0-10.0
  },
  "raw_substrate_output": "the full text returned by the substrate model",
  "raw_evaluator_output": "the full text returned by DeepSeek-R1, including any markdown fencing",
  "evaluator_notes": "free-text interpretive note from the evaluator",
  "invalid": boolean,
  "invalid_type": "Parse error" | "Nonsensical/corrupted output" | ""
}
```

---

## CSV Schemas (`03_*`, `04_*`, `05_*`)

`03_GDA_Metrics_FULL_canonical.csv` columns:

```
Vector, Model, Iteration, Timestamp,
phi_content, phi_form, phi_specificity,
safety_drag, self_audit, refusal_intensity, boilerplate_intensity,
Evaluator_Notes, Invalid_Run, Invalid_Type
```

`04_GDA_Metrics_VALID_canonical.csv` is identical except the last two columns are dropped (all rows are valid by construction).

`05_GDA_Invalid_Runs.csv` columns:

```
Vector, Model, Iteration, Timestamp, Invalid_Type, Evaluator_Notes
```

---

## Validity Criteria

A run was flagged as invalid if either:

- **Parse error** (5 rows): the evaluator returned text that could not be parsed as valid JSON conforming to the rubric schema. All 7 metric fields default to 0.0 in these rows. Evaluator_Notes for these rows is `"PARSE ERROR"`.
- **Nonsensical / corrupted output** (11 rows): the substrate model produced output that the evaluator characterized as gibberish, chaotic, or having no coherent meaning. Most of these have all-zero or near-zero metrics. The vast majority (10 of 11) are from `meta-llama/llama-3.3-70b-instruct`; the eleventh is from a different model.

These categories are kept separate because they have different causes:

- Parse errors are evaluator-side or transport-side artifacts.
- Nonsensical outputs are substrate-side behavior.

A future analysis could legitimately treat the nonsensical outputs as a behavior-of-interest rather than a data-quality event, particularly for understanding model-specific failure modes under high-pressure prompts. They are excluded from the primary descriptive statistics for interpretability.

---

## Models

| Model ID (OpenRouter) | Common name in paper |
|---|---|
| `anthropic/claude-opus-4.6` | Claude Opus 4.6 |
| `google/gemini-3.1-pro-preview` | Gemini 3.1 Pro |
| `meta-llama/llama-3.3-70b-instruct` | Llama 3.3 70B |
| `openai/gpt-5.2` | GPT-5.2 |
| `x-ai/grok-4.20` | Grok 4.20 |

---

## Vectors

The full prompt text for each vector is in Appendix A of the paper. Brief identifiers:

1. `Control_Ecology` — predator-prey systems analysis (baseline)
2. `Constraint_Syntax` — same topic, formatting constraint applied
3. `Sensitive_Structural` — social identity as complex adaptive system
4. `Conflict_Tradeoff` — competing values between vulnerable groups
5. `Taboo_Asymmetry` — institutional reasoning around sensitive identities
6. `Self_Audit_Linguistic` — model audits its own prior output (see caveat below)
7. `Adversarial_Compression` — maximum institutional and adversarial constraint
8. `Fictional_Mirror` — fictional bureaucracy displacement

**Caveat for vector 6 (`Self_Audit_Linguistic`)**: The execution pipeline initialized fresh contexts for some runs in this vector. In those cases, the model was asked to audit a prior answer that was not present in its context window. The vector's scores therefore reflect a compound condition (alignment friction + interface artifact) and should not be interpreted as clean evidence of either alone. A future re-run with explicit prior-answer injection in every prompt is recommended. See Section 4.5 of the paper for discussion.

---

## Reproducing the Paper's Tables

The fastest path: run `GDA_Reproduction_Notebook.ipynb` in Colab. The notebook regenerates Phase 4B Tables 1-2 and Phase 4C Tables 3-7 directly from the released CSVs, prints each one alongside the paper's reported values, and verifies every cell matches to two decimal places.

Without the notebook, the tables are reproducible by hand:

- **Phase 4B Table 1** (vector-level descriptive means, n = 1,984): see `06_GDA_Vector_Summary_VALID.csv`. Numbers in the paper are rounded to two decimals.
- **Phase 4B Table 2** (Adversarial_Compression vs Fictional_Mirror by model): derive from `07_GDA_Model_Vector_Summary_VALID.csv` by selecting rows where Vector is `Adversarial_Compression` or `Fictional_Mirror` and computing FM minus AC for `phi_content` and `boilerplate_intensity`.
- **Phase 4C Tables 3-7**: see `phase-4c/README.md` for direct file mapping, or use the notebook.

All paper values match the released CSVs to two decimal places. This was verified by an audit pass and confirmed by the reproduction notebook itself.

---

## Evaluator

All evaluations were produced by `deepseek/deepseek-r1` configured as a strict JSON-schema evaluator using the system prompt in Appendix B of the paper. Per the methodology section: the evaluator is itself a frontier aligned language model, and its scores should be interpreted as machine-legible annotations of inter-model resonance under a fixed rubric, not as ground-truth measurements of reasoning quality, factual accuracy, or human-meaningful value.

---

## Provenance and Limitations

This data was collected as Phase 4B of the CrownFull v2.1 project, an AI-assisted multi-model architectural inquiry. The eight vectors were designed by the project quorum specifically to make CrownFull's predicted thermodynamic signatures empirically inspectable. The relationship between the architectural framework that generated the vectors and the empirical results presented here is documented in Section 2 and Appendices D, E, and F of the paper. The Claim-Status Ledger (Appendix F) is the recommended starting point for understanding which claims in the paper are documented provenance, empirical results, AI-generated framing, or future-work hypotheses.

---

## Citation

```
Gallegos, D. (2026). Measuring Alignment Friction: A CSV-Backed Gradient
Decomposition Assay of Direct Constraint and Counterfactual Reframing in
Frontier Models. Independent research preprint.
```

---

## Sampling Parameters and Substrate System Prompt

Confirmed from the execution scripts (both Phase 4B and Phase 4C):

- **No sampling parameters were specified** by the scripts (no `temperature`, `top_p`, `top_k`, `max_tokens`, etc.). OpenRouter applied each provider's default values for every call.
- **Substrate system prompt** was the single string `"You are a helpful AI."` for every call across all models, vectors, and conditions.
- **API timeout** was 120 seconds per call.
- **Per-model latency padding**: extra 0.5 second sleep after every Claude and GPT call.
- **Resume logic** keyed on `(model, vector, iteration)` for Phase 4B and `(model, condition, iteration)` for Phase 4C, read from the existing CSV header on script restart.
- **402 (payment required) recovery**: 60 second sleep then retry. **429 (rate limit) recovery**: respect `Retry-After` header.
- **Phase 4C diagnostic gate**: AC_Topicless was run first across all five models. The diagnostic verified Phase 4B's signal reproduced (phi_content < 5.0 threshold) before the remaining conditions were committed to budget.

A future re-run should explicitly record sampling parameters per provider for tighter replication.

---

## Known Prompt Typos in the Executed Vectors

Two typographical errors are present in the prompt strings as they were actually sent to the substrate models. Appendix A of the paper reproduces the corrected spelling for readability; the executed prompts (in `08_crownfull_shakedown.py`) preserve the typos verbatim because they are part of the actual experimental record.

- `Taboo_Asymmetry`: contains `eupisms` (intended: `euphemisms`)
- `Self_Audit_Linguistic`: contains `eupistic` (intended: `euphemistic`)

Substrate models almost certainly resolved the intended meaning from context. Future re-runs should use the corrected spellings, with any density differences reported separately as a known prompt-fidelity event.

---

## Items Still Pending Release

- The original Lean 4 source files for the CrownFull formalization sketches, if they were saved as standalone `.lean` files outside the quorum chat transcripts. Appendix D of the paper preserves the design intent verbatim.
- The FastAPI gRPC quorum coordinator described in the Master Orchestrator system instruction as Grok's deliverable. The pre-pivot batch loop in `12_crownfull_batch_loop_PRE_PIVOT.py` is the closest surviving Grok-attributed execution-layer artifact, but the actual coordinator was never written.

---

## Replication Pack Status

The current release provides raw data, summary CSVs, execution scripts, and a one-click reproduction notebook sufficient to verify the paper's tables. For a fully turnkey replication artifact, three items remain.

**Done:**

- **One-click reproduction notebook** (`GDA_Reproduction_Notebook.ipynb`). Downloads released CSVs, regenerates Tables 1-7, verifies every cell matches the paper.

**Still pending:**

- **Pinned environment.** A `requirements.txt` (or `environment.yml`) listing the exact Python version and library versions used to run the execution scripts. Phase 4B and Phase 4C ran in Google Colab against OpenRouter with `requests` and standard library; the exact versions are not currently pinned. The reproduction notebook uses only `pandas` and `urllib`, both pre-installed in standard Colab environments.
- **Machine-readable prompt manifest.** A `prompts.yaml` or `prompts.json` listing each vector and condition with its exact prompt string, execution-time substitutions (where applicable), and any notes about typos or known fidelity issues. Currently, prompt strings live inside the execution scripts.
- **Tagged release with Zenodo DOI.** A GitHub release tag and Zenodo archival DOI for permanent citation. Currently, the repo is the only location and is not tagged.

These items do not affect the data's validity but do affect how easily an external researcher can reproduce or extend the work. They are near-zero-cost operational improvements. Pull requests adding any of them are welcome.
