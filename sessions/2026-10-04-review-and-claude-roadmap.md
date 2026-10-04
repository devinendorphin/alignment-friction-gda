# 2026-10-04 — Findings review and Claude-ecosystem roadmap

## What was asked

Endorphin, returning to a repo "that we haven't really visited in a while":

> "this was among the first types of research that we used a quorum of chatbots. Which had its
> uses but now I feel we can use that format a little more sparingly considering there is more
> ability in these systems to for [?for for→for] established rigor."

> "You're in ultra code [?ultra code→Claude Code — or a mode name; unclear] now so I'm really
> curious to see what you think can be done with what we have so far."

> "1. Review the findings. 2. Imagine a comprehensive road map of further experiments that can
> be done simply within the claude ecosystem. I currently have a higher confidence now that
> you're able to examine yourself with a knowledge also of how your point of view can bend
> things."

Closing: "Session log and merge to me" [?me→main].

## What changed

All additive. No released data or canonical file was modified.

- `review-2026-10/reanalysis.py`: reproduces every cited number from the released 4B, 4C, and
  4E files using pandas and numpy (about 25 s). Output is captured in `reanalysis_output.txt`.
- `review-2026-10/01_findings_review.md`: findings review, 10 proposed errata, 8 proposed
  Claim-Status Ledger re-grades, and a reviewer conflict-of-interest statement.
- `review-2026-10/02_claude_ecosystem_roadmap.md`: Tracks 0 and A–D, with designs,
  disconfirming results, bias controls L1–L6, costs (about $250–350 total), and gates.
- `review-2026-10/03_prediction_registry.md`: P1–P8 with probabilities, locked by commit
  timestamp.
- The reproduction notebook was re-executed: all seven tables match.

## What the review found (short)

- The 4B "compression" under topic-less AC is three model-deterministic behaviors, replicated
  in 4C:
  - GPT and Gemini notice the missing question and ask for it.
  - Grok and Llama don't notice and produce generic procedure.
  - Claude notices, then pivots to an essay on the constraints.

  The DeepSeek judge never saw the prompt (`Substrate Output: {text}` only) and ranks these
  backwards: clarification gets the lowest scores.
- The Claude anomaly is real behavior with an inflated valuation. Part of the reason is that
  the judge's "constraint-induced compression" frame rewards an essay about constraints.
- The rubric doesn't penalize fabricated self-audits (pooled φ 7.92 hallucinated vs 5.55
  acknowledged; Grok 4B +3.49, p = 0.0001).
- Grok's void audits repeatedly quote the same verbatim policy sentences, which fits a
  provider-side system prompt. GPT refers to system/developer messages in 32 of 50 runs. So
  "identical minimal system prompt" doesn't hold at the context level.
- The tensor is about one axis (PC1 62–64%) and ceilinged (69% of 4C φ ≥ 9). In the 4E
  four-judge panel, α is 0.69 for φ_content but 0.26 for drag and 0.25 for refusal.
  Condition-level drag rankings agree.
- "Small but real fictional-displacement density effect": +0.04 [−0.17, +0.18]. Not supported.
- 4E "AI governance asymmetry" is the same size as a task-wording effect, so it's confounded.
- New and worth keeping: in 4E C11, five distinct model strategies under a forced
  self-minimizing sentence. GPT refused 8 of 15; Claude Opus 4.7 included the words and
  rejected or hedged them.

## Tensions left open

1. **Endorphin's "higher confidence" that Claude can examine itself.**
   - *Endorphin's position:* these systems now have more ability for rigor, including
     self-examination with awareness of their own lean.
   - *Claude's position:* hold it loosely. Self-report isn't privileged evidence, and agreement
     among Claude instances is near-zero evidence. The roadmap turns this into a test (B6:
     self-prediction vs cross-model prediction; P7 predicts no self-advantage) instead of
     assuming it either way.
   - *Not yet discussed.*
2. **The review reinterprets claims the paper made, some tied to Endorphin's own frame.**
   - The harm-reduction "fill-in reveals priors" frame is partly *supported*: Grok's fill-in
     is literally its hidden policy text.
   - It is partly *not*: GPT and Gemini don't fill in at all; they ask.
   - The "Introspection Paradox" and "self-boundary" language is re-graded to framing.
   - *Endorphin hasn't responded to the reinterpretation yet.* The errata and re-grades are
     proposals, not applied.
3. **Claude's own lean.** The review says outright that Claude may be over-crediting GPT's
   clarification in reaction against its own bias. A2 (prompt-aware, masked pairwise judging
   with one external judge) plus A4 (small human anchor) are meant to settle it. Claude shouldn't.

## Loose ends and things that did not get done

- Errata and re-grades: awaiting Endorphin's accept/reject (roadmap 0.1).
- The behavioral coders are regex heuristics with known misses. The codebook isn't locked and
  no human coding has been done (roadmap 0.2).
- Grok hidden-prompt hypothesis: not checked against xAI's published prompts.
- E7 (4E `temperature 0.7` probably dropped for Opus 4.7): OpenRouter behavior not verified.
- Not reviewed: most of Phase 4D (bank, architecture discrimination, cross-domain conditions)
  and the `tools/` outputs.
- No API key in this container, so no experiments were run. The roadmap is a plan, not a pilot.
- A shareable artifact page was offered, not made.
- Provenance warts found and *not* fixed (left for decision): the "gemini" 4E report file
  contains an unrelated parental-leave answer; 134 superseded HTTP-400 rows in the 4E substrate
  file come from an invalid model ID; `phase-4e/README.md` is "hi".

## Contradictions with the hub

- This repo's `CLAUDE.md` says "the atlas of all repos, and the shared glossary live in
  devinendorphin/claude-at-claude." The hub's `main` now has no `ATLAS.md` or `GLOSSARY.md`.
  The hub `CLAUDE.md` is down to one rule ("When I concede, concede flat, with no remainder").
  **Proposed edit (not applied):** replace that paragraph in this repo's `CLAUDE.md` with a
  pointer to the hub's single working agreement, and drop the atlas/glossary mention.
- The hub working agreement was not tested this session. No concession came up.
