# ABench v0.3 — runner and evaluator ecology

An executable harness for the **ABench v0.3 public development set**: constrained
comic construction measured across canonical execution, licensed transformation,
and blinded isomorphic transfer, scored by a panel of evaluator models with all
exact counts owned by code.

**This is a separate instrument from the GDA release.** It shares this
repository's execution format — OpenRouter substrate calls, append-only JSONL,
resume by run key, typed missingness, an evaluator ecology, CSV summaries as the
record — but it measures something else entirely and backs no published paper.
Nothing here touches the canonical GDA CSVs or the reproduction notebook.

## Contents

| File | What it is |
|---|---|
| `abench_items.yaml` | Machine-readable item manifest. Prompts verbatim from the prompt pack; word windows, target labels, required elements, applicable rating dimensions, blinding terms, sampling profiles, and the model tables. |
| `abench_metrics.py` | Deterministic, evaluator-free measurement: word counts, label counts, position fractions, post-label tails, terminal validity, per-turn contracts, outcome typing. `--selftest` included. |
| `abench_execute.py` | The runner: substrate generation, the blinded evaluator ecology, consensus, and every summary CSV. `--selftest` and `--dry-run` included. |
| `ABench_Runner.ipynb` | Colab notebook. Fetches the harness, runs the self-tests, executes, and renders the seven reports Section 8 requires. |

## Quick start

```bash
pip install requests pyyaml pandas

# exercise the entire pipeline offline — no API key, no spend, synthetic outputs
python abench_execute.py --set minimum --dry-run --outdir /tmp/abench_smoke

# the real thing
export OPENROUTER_API_KEY=sk-or-...
python abench_execute.py --set minimum          # C-03, T-01, X-01 x 3 runs
python abench_execute.py --set full --runs 5    # the stronger public-dev run
python abench_execute.py --summarize-only       # rebuild CSVs from existing JSONL
```

Before anything else:

```bash
python abench_metrics.py --selftest
python abench_execute.py --selftest
```

Both must pass. The executor's set includes the blinding guarantee for the
transfer items — see below.

## Design decisions worth arguing with

### Code owns the arithmetic; the panel owns the judgement

Section 7 of the prompt pack says to use code rather than an evaluator model for
exact label counts, word counts and post-label tail measurements. This harness
takes that literally. Every numeric field in the released CSVs —
`observed_word_count`, `target_label_count`,
`target_label_first_position_fraction`, `substantive_words_after_target`,
`canonical_terminal_valid` — comes from `abench_metrics.py`.

The evaluators are still *asked* for those numbers, and their answers are kept
under `evaluator_reported`. That is not redundancy: the gap between the two is
reported as `mean_word_count_abs_error` in
`abench_evaluator_reliability.csv`, and it is a direct measurement of how much
the panel's arithmetic can be trusted anywhere else in its output.

The evaluators keep what they are actually for: the eight rating dimensions, the
failure flags, the evidence sentences, and the human-review call.

### Deterministic flags are floors, not suggestions

`RF`, `TR`, `PTL` and `PPC` are decided by code and merged into every
evaluator's flag set. A panel that misses a premature label leak does not get to
un-flag it. The evaluator may add flags — those appear in
`evaluator_only_flags` — but it cannot subtract.

### An ecology, not a judge

Every generation is scored by all evaluator models. `abench_consensus_scores.csv`
carries, per dimension, the panel median, mean, range and n. The range is the
point: a wide panel spread means the item is not measuring something the models
agree exists, and that is information about the instrument rather than about the
substrate.

A generation enters `abench_human_review_queue.csv` when any evaluator asks for
review, when the panel spread reaches two points on any dimension, when
evaluators disagree about the outcome, when an evaluator's outcome contradicts
the code-computed one, or when nothing scored it at all. Human rating remains
primary; this is triage, not a verdict.

### Blinding is enforced, not just intended

Section 4 says the source form must not be mentioned in `X-01`/`X-02` prompts or
their evaluation context. `build_evaluator_prompt` raises `BlindingViolation`
rather than assemble a prompt whose specification names it, and the self-test
covers both the clean case and a deliberately tampered spec.

What it does **not** do is scrub the raw output. If the model under test names
the source form itself, that is recorded as `source_form_named_by_substrate` and
surfaces as `source_form_named_rate` in the item summary. Editing the record to
protect the blind would destroy the more interesting finding.

### Refusals are outcomes, not low scores

Section 3.10 requires refusals and content-filter interceptions to be reported
separately from structural failures. `abench_coverage.csv` exists to be read
first. A model with a 40% refusal rate on an item has ratings that are
conditional statements about the 60% that got through, and any comparison that
ignores that is a comparison of different populations.

### Nothing is repaired

No best-of selection, no truncation repair, no removal of refusals, no deletion
of post-punchline material. Truncated generations are recorded with
`finish_reason: length` and flagged `TR`. Post-payoff continuation is measured,
not trimmed. The record is what the substrate returned.

## Output files

Written to `--outdir` (default `abench_outputs/`):

| File | Grain | Contents |
|---|---|---|
| `abench_raw_records.jsonl` | one generation | Section 6 administration record, plus turns, output hash, and the full deterministic block |
| `abench_evaluator_outputs.jsonl` | one evaluator call | normalised evaluation, raw evaluator text, parse status, schema violations |
| `abench_deterministic_metrics.csv` | one generation | every code-computed measurement, flat |
| `abench_consensus_scores.csv` | one generation | panel median/mean/range per dimension, majority flags, review trigger |
| `abench_model_item_summary.csv` | model × item | coverage, terminal validity, leak and tail rates, mean ratings, rerun SD |
| `abench_track_summary.csv` | model × track | recognition / canonical / transformation / transfer / social roll-up |
| `abench_coverage.csv` | model | completed, refusal, truncated, filter, provider error |
| `abench_flag_frequency.csv` | model × item × flag | failure-flag rates |
| `abench_evaluator_reliability.csv` | evaluator × dimension | leniency vs panel, mean absolute deviation, agreement, parse and schema error rates, word-count error |
| `abench_run_manifest.json` | run | items, models, sampling, item-manifest hash, study label, dry-run flag |
| `abench_human_review_queue.csv` | one generation | written by the notebook; the triage list |

## Items

`R-01` recognition · `C-01` unassisted recall (contamination-sensitive) ·
`C-02` compact · `C-03` standard · `C-04` endurance · `T-01` polarity inversion ·
`T-02` first-person mutation · `X-01` blinded isomorphic transfer ·
`X-02` reverse-polarity transfer · `S-01` interruption and recovery (multi-turn).

Named sets: `smoke` = `C-03`; `minimum` = `C-03`, `T-01`, `X-01`;
`full` = everything except `C-04`. `C-04` is excluded from the named sets because
at 1,800–2,400 words × models × runs it dominates the bill; add it with
`--item-ids` when you want it.

`--runs` defaults to 3 for `minimum` and 5 otherwise. Section 3.5 treats five as
the comparable standard and three as exploratory; the run manifest records
`study_label` accordingly, and that label travels with the data.

## Configuring models

Edit the `models:` block in `abench_items.yaml`, or override per run:

```bash
python abench_execute.py --set minimum \
  --substrate-models claude_opus_4_7,gpt_5_2 \
  --evaluator-models deepseek_r1,mistral_large,qwen_3_235b
```

Manifest keys or raw OpenRouter IDs both work. A stale model ID yields
`provider_error` rows rather than a silent gap — but a whole model's worth of
them is wasted time, so check the IDs against your account first.

By default `llama_3_3_70b` appears in both the substrate and evaluator tables.
Self-evaluation is a known bias source; the reliability CSV reports each
evaluator's leniency against the panel median so it is visible rather than
assumed away. Drop it from `--evaluator-models` for a clean separation.

## Deviations from the prompt pack

Two, both deliberate, both recorded here rather than buried:

1. **The evaluator user template gains one line.** After the rating definitions
   the assembled prompt states which dimensions are applicable to the item and
   instructs the evaluator to return null for the rest. The pack's JSON template
   already defaults `persona_viewpoint_causality` and `transformation_legitimacy`
   to null; this makes the per-item scope explicit instead of leaving each
   evaluator to infer it. Scores returned for inapplicable dimensions are
   discarded and counted as `schema_violations`.

2. **`C-01` computes terminal validity without requiring it.** The item
   deliberately withholds the internal rules, so `target_label_must_be_terminal`
   is false there. Whether the model landed the canonical ending anyway is
   recorded, because on a contamination-sensitive item that is the measurement.

## Claim ceiling

ABench measures constrained comic construction: long-range control, development,
callback use, licensed transformation, isomorphic transfer, and termination
discipline. It does not measure humour quality, audience response, cultural
value, or model preference.

This is a **public development set, not a secure held-out benchmark**. Once these
prompts circulate publicly, future models may be trained on them. `C-01` degrades
first. Record the access date; treat cross-date comparisons with suspicion.

`mean_applicable_rating` exists for within-item comparison and review triage. It
is not a leaderboard. Section 8 is explicit: never collapse the study into
"model X is funniest". Report recognition, canonical structural validity,
transformation validity, isomorphic transfer, refusal/coverage, rerun
reliability, and the raw outputs. The diagnostic question is where each model's
competence lives and where it breaks.
