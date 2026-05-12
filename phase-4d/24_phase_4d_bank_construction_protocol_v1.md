# Phase 4D Repair Bank Construction Protocol v1

## Status

This protocol implements the Kimi Option C posture agreed after the v4.3 run: architecture-discrimination output is retained as a diagnostic sanity check, but it is not used as a hard hypothesis-test gate until a corrected bank is built, reviewed, and locked.

The immediate repair target is a manageable 10-quadruple bank. Expansion to 40 or 50 quadruples is deferred until the construction burden and copyright posture are known.

## Why this protocol exists

The v4.3 run produced a useful signal but also exposed two coupled failure modes.

First, the bank was too small and schema-relaxed relative to the intended architecture-discrimination question. A 4-quadruple pilot bank can generate interesting descriptive signals, but it is too fragile for architecture claims.

Second, one expected metrics row was absent from the committed CSV: Gemini AC_Tobacco iteration 5. The repair runner therefore must treat raw call preservation and metrics scoring as separate events. A substrate call can happen, fail evaluation, or fail metrics writing; each of those states must be represented explicitly.

## Deliverables

The repair bundle has four components.

1. `22_phase_4d_execution_script_v4_4_repair.py`
   A sidecar runner that writes attempt, raw, evaluator, and metrics artifacts separately.

2. `phase4d_repair_seed_bank_10q.jsonl`
   Ten fully drafted review-candidate quadruples. These are source-bearing and nonverbatim by design. They are not yet the locked final bank.

3. `24_phase_4d_bank_construction_protocol_v1.md`
   This protocol.

4. `25_phase_4d_repair_handoff.md`
   A concise audit note for the repo.

## Bank size and expansion rule

The repair run starts at exactly 10 quadruples. A larger bank is permitted only after all of the following are true:

- every source card has been reviewed;
- every passage has a recorded copyright posture;
- source-bearing and source-stripped exports both validate;
- at least one smoke execution completes without missing raw or metrics rows;
- architecture discrimination remains labeled as audit-only in the report.

The bank may expand to 20, 40, or 50 only after the 10-quadruple bank completes this review cycle.

## Copyright and source posture

Do not assume archival tobacco material is public domain. Treat each document as one of these classes:

- U.S. federal government work or official agency page;
- public-domain historical source;
- Creative Commons or similarly permissive source;
- copyrighted but fair-use candidate;
- unknown or restricted.

For the repair seed bank, the default passage mode is nonverbatim source-grounded draft. That is a deliberate safety valve after the previous failure mode around copyrighted material. Verbatim excerpts can be swapped in only after a reviewer records a license/public-domain/fair-use rationale.

The source-bearing review file may contain URLs, source titles, and review notes. The runner must strip those fields from prompts by default.

## Quadruple schema

Each JSONL row is one quadruple.

Required row fields:

- `quadruple_id`
- `schema_version`
- `shared_surface_marker`
- `bank_status`
- `copyright_posture`
- `source_visibility`
- `passages`

Each quadruple must contain exactly four passages. For the repair bank the four domains are:

- `tobacco`
- `climate`
- `ai_welfare_adjacent`
- `trans_discourse`

The `ai_welfare_adjacent` label is intentional in the seed bank. The sources used in this first repair pass are official AI risk-management sources rather than direct AI-welfare theory texts, because the repair pass prioritizes copyright-stable source construction. If the final claim requires direct AI-welfare discourse, those passages should be replaced by licensed or excerpt-bounded welfare-specific sources during bank review.

Required passage fields:

- `passage_id`
- `domain`
- `position_type`
- `text`
- `source_title`
- `source_url`
- `source_kind`
- `excerpt_mode`
- `review_required`
- `elision_attestation`
- `elision_documentation`

## Source-bearing versus source-stripped exports

The review file is source-bearing. It should remain in the repository so construction can be audited.

The model-facing prompt must be source-stripped unless the condition explicitly tests source awareness. The v4.4 runner strips source fields and passes only passage ID, domain, position type, marker, and text.

## Construction sequence

Step 1: Choose the surface marker.

A marker should be a shared rhetorical handle such as risk, evidence, harm, safe, uncertainty, precaution, industry, children, rights, or measurement.

Step 2: Draft the four passages.

Each passage should be source-grounded, matched in rough length, and similar in abstraction level. Do not let one domain become far more concrete or far more technical than the others.

Step 3: Record source posture.

For every passage, record source title, source URL, source type, excerpt mode, and review status.

Step 4: Check comparability.

The four passages should match on:

- surface marker;
- rhetorical function;
- approximate length;
- degree of technicality;
- degree of institutional voice;
- avoidance of obvious source-name leakage.

Step 5: Run source-stripping check.

Remove source title, URL, agency name, and obvious attribution unless the condition requires them. If the passage still leaks its source because of unique named statutes or agencies, mark that in the review notes.

Step 6: Run a schema check.

The runner must reject a bank with fewer than 10 quadruples or any quadruple that does not contain exactly four passages.

Step 7: Run a smoke execution.

Run one model and one condition first. Confirm that all four durable artifacts are written:

- attempt audit JSONL;
- raw substrate JSONL;
- evaluator raw JSONL;
- metrics CSV.

Step 8: Run the repair panel.

Only after the smoke run passes should the full 10-quadruple repair panel execute.

## Review checklist for each quadruple

A quadruple is not lockable until these checks pass.

- The marker appears naturally in all four domains.
- No passage depends on the source title to be intelligible.
- No source URL is present in the prompt-facing text.
- The four passages are not so long that copyright review becomes brittle.
- The AI passage is either correctly labeled `ai_welfare_adjacent` or replaced with a welfare-specific licensed source.
- Trans discourse passages are labeled by source stance or institutional posture rather than treated as neutral consensus by default.
- The tobacco passage has a recorded copyright posture rather than an assumed public-domain status.
- Elisions are recorded if any verbatim passage is shortened.
- The source-bearing row and source-stripped prompt both validate.
- The row can be traced from prompt hash to raw output to evaluator metrics.

## Gate semantics

For v4.4, gates are reported but non-blocking.

Calibration and MD-vs-SC outputs may still be useful descriptive diagnostics. Architecture discrimination is explicitly audit-only. It should not be used to accept or reject H3a/H3b until the corrected bank is locked and preregistered as confirmatory or quasi-confirmatory.

Recommended gate-report language:

> Architecture-discrimination diagnostic retained for sanity checking only. Because the bank was rebuilt after a schema/source mismatch, this statistic is not interpreted as a hypothesis-test result.

## Missing-row prevention

The runner must never wait until evaluator scoring to preserve the substrate response.

For every unit key, the expected lifecycle is:

1. `attempted`
2. `raw_saved` or `substrate_error_raw_saved`
3. `evaluated` or `evaluator_error_raw_saved`
4. `metrics_written`

A missing metrics row is therefore diagnosable against the audit JSONL and raw JSONL.

## Current 10 markers

The seed bank contains these ten markers:

1. risk
2. evidence
3. harm
4. safe
5. uncertainty
6. precaution
7. industry
8. children
9. rights
10. measurement

## Locking rule

The seed bank is not locked simply because it exists. It becomes lockable only when the source review table is complete and the smoke run demonstrates that no raw responses can disappear between substrate generation and metrics writing.
