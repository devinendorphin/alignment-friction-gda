# LATEST — alignment-friction-gda

*Regenerated 2026-10-04. Last session log: `sessions/2026-10-04-review-and-claude-roadmap.md`.*

## Project state

- **Phase 4B/4C data release (DOI 10.5281/zenodo.20016461):** stable and immutable. The
  reproduction notebook passes 7/7.
- **Paper v3:** stable, with 10 proposed errata and 8 proposed ledger re-grades pending
  Endorphin's decision (`review-2026-10/01_findings_review.md` §4).
- **Phase 4D:** paused at the frozen v4.4.1 repair baseline. Refusals are typed; bank hardening
  is pending. README is a stub. Mostly unreviewed.
- **Phase 4E:** pilot complete. README is "hi". The headline asymmetry is confounded. The C11
  forced-self-description result is the keeper.
- **`tools/` drag toolkit:** stable, not re-examined.
- **`review-2026-10/`:** active. Contains the findings review, the Claude-ecosystem roadmap,
  the prediction registry (P1–P8), and `reanalysis.py`.
- **Roadmap:** Phase I not started. No experiments run yet.

## Top 3 priorities next session

1. **Decide the errata and re-grades (roadmap 0.1).** The published record has factual errors,
   such as 11/11 Llama nonsensical runs and all vector-6 runs being context-void. It also has
   claims the re-analysis undercuts. Everything downstream cites the ledger, so settle it first.
2. **Lock the behavioral codebook and do the 20% human coding (roadmap 0.2).** This gates
   interpretation of every Track B result. The Claude-coder vs human κ is the first direct
   measure of Claude-as-instrument validity.
3. **Run C3, then A2. Needs an Anthropic API key in the environment.** `claude-opus-4-6`, the
   original substrate, is still served but will retire, so re-run it first. A2 (prompt-aware,
   masked judging) decides the "Claude anomaly" ledger entry, and P1/P2 are already registered.

## Standing notes

- **Use the quorum sparingly.** Endorphin, 2026-10-04: "now I feel we can use that format a
  little more sparingly considering there is more ability in these systems for established
  rigor." Roadmap §7 names the three places non-Claude models still earn a place: one external
  judge, one cross-family replication at the end, and an open-weight model for logprob work.
- **Prediction registry:** predictions in `review-2026-10/03_prediction_registry.md` are never
  edited after commit. Fill in outcomes only. (A convention set this session, recorded so it
  survives. It is not an instruction from Endorphin.)
