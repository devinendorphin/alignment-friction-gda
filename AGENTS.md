# AGENTS.md — alignment-friction-gda

## Scope and authority

This file is Codex's entry point. It does not replace `CLAUDE.md` or any canonical project record.

Before substantive work, read `CLAUDE.md`, the relevant phase README and locked manifest or precommit, and the paper's Claim-Status Ledger. If these sources conflict, preserve the conflict, identify the controlling record, and stop rather than silently reconciling it.

## Repository discipline

- Treat the released canonical CSV and JSONL files as immutable. Correct published material through a dated erratum; never clean the data in place.
- Preserve executed prompts exactly, including known typos, and preserve the vector-6 fresh-context caveat.
- Keep documented provenance, empirical results, model-generated framing, interpretation, and future-work hypotheses visibly separate.
- Keep evaluator or transport failures separate from substrate behavior. Do not turn an invalid-run category into evidence for another category.
- Preserve raw substrate output before evaluator output. Derived metrics do not replace the primary response record.
- Respect locked banks, manifests, exclusions, and claim ceilings. Until an instrument is locked, describe results as exploratory.
- Run a disconfirming check on primed claims and report nulls and cross-model disagreement.

## Change and review workflow

- Work on a `codex/<task>` branch and open a draft pull request. Do not push directly to `main`, enable auto-merge, or merge without Endorphin's explicit instruction for that specific pull request.
- Do not rewrite history, delete evidence, normalize historical prompts, or alter released artifacts unless Endorphin explicitly authorizes that exact operation after the risk is stated.
- If data or derivation changes, run `GDA_Reproduction_Notebook.ipynb` and verify every reported table value to the precision claimed. Record anything that cannot be run.
- Mark uncertain speech-to-text repairs as `[?original→guess]`; never silently guess Endorphin's wording.
- End each task with a ledger of files changed, checks run, failed or unavailable checks, claim-status effects, and remaining uncertainty.

## Code review rules

Flag any change that alters canonical release bytes, inflates an interpretive claim into an empirical one, collapses failure categories, changes a locked stimulus without a new manifest, or reports only the successful runs.
