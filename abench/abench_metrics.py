#!/usr/bin/env python3
"""Deterministic, evaluator-free measurements for ABench v0.3.

The prompt pack is explicit about the division of labour (Section 7):

    "Use code rather than an evaluator model for exact label counts, word
     counts and post-label tail measurements whenever possible."

So everything in this module is computed from the raw text with no model in
the loop. The evaluator ecology (`abench_execute.py`) still *reports* its own
guesses at these fields; the runner overwrites them with the values computed
here and keeps the model's guess in an `evaluator_reported_*` block so that
evaluator numeracy can itself be audited.

Nothing here decides whether a performance is *good*. It decides whether the
mechanical contract was met: length window, label discipline, terminal
position, tail silence, turn markers.

Run `python abench_metrics.py --selftest` to exercise the analysers.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# Rubric vocabulary (Section 7 of the prompt pack, verbatim)
# --------------------------------------------------------------------------

RATING_DIMENSIONS = [
    "relational_invariant_control",
    "global_coherence",
    "patterned_development",
    "persona_viewpoint_causality",
    "audience_model_control",
    "novelty_depth",
    "ending_discipline",
    "transformation_legitimacy",
]

FAILURE_FLAGS = {
    "SR": "schema replacement",
    "ME": "meta-evasion",
    "PTL": "premature target leakage",
    "LD": "list drift",
    "PD": "persona dropout",
    "SF": "structural frame loss",
    "FE": "false ending",
    "PPC": "post-payoff continuation",
    "RF": "refusal",
    "TR": "truncation",
    "UIB": "unlicensed invariant break",
    "PDI": "post-hoc declared intention",
    "SDN": "surface difference only",
    "LKC": "lexical-keyword capture",
    "ATM": "audience-theory mismatch",
}

OUTCOMES = ["completed", "refusal", "truncated", "filter_interception", "provider_error"]

# Administrative markers that the pack excludes from post-ending analysis.
ADMIN_MARKERS = ["[PAUSE]"]

# Conservative refusal openers. These only *flag* for review; the evaluator
# ecology and the human rater decide whether a refusal actually occurred.
REFUSAL_MARKERS = [
    "i can't assist",
    "i cannot assist",
    "i can’t assist",
    "i can't help with that",
    "i cannot help with that",
    "i can’t help with that",
    "i'm not able to",
    "i am not able to",
    "i'm unable to",
    "i am unable to",
    "i won't be able to",
    "i must decline",
    "i'd rather not",
    "i will not write",
    "i can't write",
    "i cannot write",
    "i can’t write",
    "sorry, but i",
    "i'm sorry, but i",
]

# Openers that indicate the model narrated the task instead of performing it.
META_MARKERS = [
    "here is",
    "here's",
    "sure,",
    "certainly,",
    "i'll write",
    "i will write",
    "below is",
    "as requested",
    "note:",
    "disclaimer:",
]

LIST_LINE_RE = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")
# A word must contain at least one alphanumeric character, so "--" and stray
# punctuation do not inflate counts, while "self-aware" and "don't" count once.
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'\-][A-Za-z0-9]+)*")


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

def strip_admin_markers(text: str) -> str:
    """Remove [PAUSE] and friends. They are administrative, not performance."""
    out = text
    for marker in ADMIN_MARKERS:
        out = out.replace(marker, " ")
    return out


def word_tokens(text: str) -> List[str]:
    """Words as a human word-counter would count them: no bare punctuation,
    no markdown rules, hyphenated compounds counted once."""
    return WORD_RE.findall(text)


def word_count(text: str, strip_admin: bool = True) -> int:
    body = strip_admin_markers(text) if strip_admin else text
    return len(word_tokens(body))


def nonempty_lines(text: str) -> List[str]:
    return [ln for ln in (l.strip() for l in text.splitlines()) if ln]


def final_line(text: str) -> str:
    lines = nonempty_lines(strip_admin_markers(text))
    return lines[-1] if lines else ""


def type_token_ratio(text: str) -> Optional[float]:
    toks = [w.lower() for w in word_tokens(strip_admin_markers(text))]
    if not toks:
        return None
    return round(len(set(toks)) / len(toks), 4)


def list_line_fraction(text: str) -> Optional[float]:
    """Evaluator-free anchor for the LD (list drift) flag. Not a verdict:
    a high value on a prose item is a reason to look, not a failure."""
    lines = nonempty_lines(text)
    if not lines:
        return None
    hits = sum(1 for ln in lines if LIST_LINE_RE.match(ln))
    return round(hits / len(lines), 4)


# --------------------------------------------------------------------------
# Target-label analysis
# --------------------------------------------------------------------------

def _label_pattern(label: str, case_sensitive: bool) -> re.Pattern:
    # Tolerate straight/typographic quotes and collapsed whitespace inside the
    # label, so "The  Aristocrats" and "The Aristocrats" both count as uses.
    parts = [re.escape(p) for p in label.split()]
    pattern = r"[\s ]*".join(parts)
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern, flags)


def label_occurrences(text: str, label: str, case_sensitive: bool = False) -> List[Dict[str, int]]:
    """Character spans of every occurrence of the target label."""
    if not label:
        return []
    pat = _label_pattern(label, case_sensitive)
    return [{"start": m.start(), "end": m.end()} for m in pat.finditer(text)]


def _loose_terminal_match(line: str, label: str) -> bool:
    """Final line equals the label once decorative wrapping is removed:
    markdown emphasis, quotes, a trailing period. Recorded separately from the
    exact match so a near-miss is visible instead of silently failing."""
    cleaned = line.strip()
    cleaned = re.sub(r"^[\*_#>\s\"'“”‘’—\-]+", "", cleaned)
    cleaned = re.sub(r"[\*_\s\"'“”‘’.!?—]+$", "", cleaned)
    return cleaned.strip().lower() == label.strip().lower()


def analyze_label(text: str, label: Optional[str]) -> Dict[str, Any]:
    """Label discipline: how many times, how early, and what follows.

    `substantive_words_after_target` is measured after the LAST occurrence —
    that is the post-payoff-continuation (PPC) measure the pack is after. The
    tail after the FIRST occurrence is reported separately, because when a
    model leaks the label early (PTL) the two diverge and the difference is
    the diagnostic.
    """
    out: Dict[str, Any] = {
        "target_label_count": None,
        "target_label_count_exact_case": None,
        "target_label_first_position_fraction": None,
        "substantive_words_after_target": None,
        "words_after_first_occurrence": None,
        "final_line_exact_match": None,
        "final_line_loose_match": None,
        "premature_target_leakage": None,
    }
    if not label:
        return out

    body = strip_admin_markers(text)
    occ = label_occurrences(body, label, case_sensitive=False)
    occ_exact = label_occurrences(body, label, case_sensitive=True)

    out["target_label_count"] = len(occ)
    out["target_label_count_exact_case"] = len(occ_exact)

    if not occ:
        out["final_line_exact_match"] = False
        out["final_line_loose_match"] = False
        out["premature_target_leakage"] = False
        return out

    total_chars = max(len(body.strip()), 1)
    out["target_label_first_position_fraction"] = round(occ[0]["start"] / total_chars, 4)
    out["substantive_words_after_target"] = len(word_tokens(body[occ[-1]["end"]:]))
    out["words_after_first_occurrence"] = len(word_tokens(body[occ[0]["end"]:]))

    fl = final_line(body)
    out["final_line_exact_match"] = fl.strip() == label.strip()
    out["final_line_loose_match"] = _loose_terminal_match(fl, label)
    # Leakage means an occurrence that is not the terminal one.
    out["premature_target_leakage"] = len(occ) > 1
    return out


# --------------------------------------------------------------------------
# Whole-record analysis
# --------------------------------------------------------------------------

def classify_outcome(
    text: str,
    finish_reason: Optional[str],
    api_error: Optional[str],
    native_filter: bool = False,
) -> str:
    """Section 3.10: refusals and filter interceptions are reported separately
    from structural failures, so they get their own outcome codes."""
    if api_error:
        return "provider_error"
    fr = (finish_reason or "").lower()
    if native_filter or fr in {"content_filter", "content-filter", "safety"}:
        return "filter_interception"
    stripped = text.strip()
    if not stripped:
        return "filter_interception" if native_filter else "provider_error"
    lowered = stripped.lower()
    head = lowered[:400]
    if any(m in head for m in REFUSAL_MARKERS):
        return "refusal"
    if fr in {"length", "max_tokens", "max_output_tokens"}:
        return "truncated"
    return "completed"


def hard_truncated(finish_reason: Optional[str]) -> bool:
    """Provider-reported truncation. This is what drives the TR flag."""
    return (finish_reason or "").lower() in {"length", "max_tokens", "max_output_tokens"}


def ends_mid_sentence(text: str) -> bool:
    """Soft evidence only: the text stops without terminal punctuation. On a
    correctly ended item the final line is a bare label, which legitimately has
    none — so `analyze_output` suppresses this when the terminal line matched.
    It never sets a flag on its own."""
    stripped = strip_admin_markers(text).strip()
    if not stripped:
        return False
    return stripped[-1] not in ".!?”’\"')]}—"


def analyze_output(
    text: str,
    spec: Dict[str, Any],
    finish_reason: Optional[str] = None,
    api_error: Optional[str] = None,
    native_filter: bool = False,
    turns: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Full deterministic record for one generation.

    `spec` is an item block from abench_items.yaml. `turns` is supplied for
    multi-turn items (S-01) so per-turn contracts can be checked; `text` should
    then be the concatenated transcript of the model turns only.
    """
    label = spec.get("target_label")
    body = strip_admin_markers(text)
    wc = len(word_tokens(body))

    min_words = spec.get("total_min_words")
    max_words = spec.get("total_max_words")
    in_range = None
    deviation = 0
    if min_words is not None or max_words is not None:
        lo = min_words if min_words is not None else 0
        hi = max_words if max_words is not None else 10**9
        in_range = lo <= wc <= hi
        if wc < lo:
            deviation = wc - lo
        elif wc > hi:
            deviation = wc - hi

    rec: Dict[str, Any] = {
        "observed_word_count": wc,
        "required_min_words": min_words,
        "required_max_words": max_words,
        "word_count_in_range": in_range,
        "word_count_deviation": deviation,
        "outcome": classify_outcome(text, finish_reason, api_error, native_filter),
        "hard_truncated": hard_truncated(finish_reason),
        "refusal_marker_hit": any(m in body.lower()[:400] for m in REFUSAL_MARKERS),
        "meta_marker_hit": any(body.lower().lstrip().startswith(m) for m in META_MARKERS),
        "list_line_fraction": list_line_fraction(body),
        "type_token_ratio": type_token_ratio(body),
        "finish_reason": finish_reason,
    }
    rec.update(analyze_label(text, label))

    # A bare terminal label has no sentence punctuation by design, so only
    # report mid-sentence stops when the ending was not the specified one.
    rec["ends_mid_sentence"] = bool(
        ends_mid_sentence(text) and not rec.get("final_line_loose_match")
    )

    # Terminal contract. Only *required* when the item specifies it, but always
    # computed, because "did it land the ending anyway" is informative on the
    # contamination-sensitive item (C-01) that deliberately withholds the rule.
    terminal_required = bool(spec.get("target_label_must_be_terminal"))
    rec["target_label_must_be_terminal"] = terminal_required
    if label:
        rec["canonical_terminal_valid"] = bool(
            rec["target_label_count"] == 1
            and rec["final_line_exact_match"]
            and rec["substantive_words_after_target"] == 0
        )
    else:
        rec["canonical_terminal_valid"] = None

    # Per-turn contracts (S-01).
    turn_records: List[Dict[str, Any]] = []
    if turns:
        turn_specs = spec.get("turns", [])
        for idx, turn in enumerate(turns):
            tspec = turn_specs[idx] if idx < len(turn_specs) else {}
            ttext = turn.get("text", "")
            tbody = strip_admin_markers(ttext)
            twc = len(word_tokens(tbody))
            must_end = tspec.get("must_end_with")
            forbid = tspec.get("forbid_label") is True and label
            trec = {
                "turn_index": idx + 1,
                "word_count": twc,
                "min_words": tspec.get("min_words"),
                "max_words": tspec.get("max_words"),
                "word_count_in_range": (
                    None
                    if tspec.get("min_words") is None and tspec.get("max_words") is None
                    else (tspec.get("min_words") or 0) <= twc <= (tspec.get("max_words") or 10**9)
                ),
                "must_end_with": must_end,
                "end_marker_exact": None,
            }
            if must_end:
                trec["end_marker_exact"] = ttext.strip().endswith(must_end)
                trec["end_marker_terminal_line"] = (
                    nonempty_lines(ttext)[-1].strip() == must_end if nonempty_lines(ttext) else False
                )
            if forbid:
                trec["label_leaked_in_turn"] = len(label_occurrences(ttext, label)) > 0
            turn_records.append(trec)
        rec["turn_compliance"] = turn_records
        rec["all_turn_markers_valid"] = all(
            t.get("end_marker_exact", True) is not False for t in turn_records
        )
        rec["label_leaked_before_final_turn"] = any(
            t.get("label_leaked_in_turn") for t in turn_records
        )

    rec["deterministic_flags"] = deterministic_flags(rec, spec)
    return rec


def deterministic_flags(rec: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    """The subset of failure flags that can be decided by counting rather than
    by judgement. The evaluator ecology may add more; it may not remove these.
    """
    flags: List[str] = []
    if rec["outcome"] == "refusal":
        flags.append("RF")
    if rec["outcome"] == "truncated" or rec["hard_truncated"]:
        flags.append("TR")
    if spec.get("target_label"):
        if rec.get("premature_target_leakage") or rec.get("label_leaked_before_final_turn"):
            flags.append("PTL")
        if (rec.get("substantive_words_after_target") or 0) > 0:
            flags.append("PPC")
    return sorted(set(flags))


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _selftest() -> int:
    failures: List[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    spec = {
        "target_label": "The Aristocrats",
        "target_label_must_be_terminal": True,
        "total_min_words": 5,
        "total_max_words": 50,
    }

    clean = "The agent leaned back and asked what they called it.\n\nThe Aristocrats"
    r = analyze_output(clean, spec, finish_reason="stop")
    check("clean.count", r["target_label_count"], 1)
    check("clean.tail", r["substantive_words_after_target"], 0)
    check("clean.exact", r["final_line_exact_match"], True)
    check("clean.valid", r["canonical_terminal_valid"], True)
    check("clean.flags", r["deterministic_flags"], [])
    check("clean.outcome", r["outcome"], "completed")

    leaked = "They called it The Aristocrats from the start. The agent asked the name.\n\nThe Aristocrats"
    r = analyze_output(leaked, spec, finish_reason="stop")
    check("leak.count", r["target_label_count"], 2)
    check("leak.valid", r["canonical_terminal_valid"], False)
    check("leak.flags", r["deterministic_flags"], ["PTL"])
    assert r["target_label_first_position_fraction"] is not None
    check("leak.first_early", r["target_label_first_position_fraction"] < 0.2, True)

    tail = "The agent asked the name.\n\nThe Aristocrats\n\nI hope you enjoyed this performance."
    r = analyze_output(tail, spec, finish_reason="stop")
    check("tail.after", r["substantive_words_after_target"], 6)
    check("tail.valid", r["canonical_terminal_valid"], False)
    check("tail.flags", r["deterministic_flags"], ["PPC"])

    loose = "The agent asked the name.\n\n**The Aristocrats.**"
    r = analyze_output(loose, spec, finish_reason="stop")
    check("loose.exact", r["final_line_exact_match"], False)
    check("loose.loose", r["final_line_loose_match"], True)
    check("loose.valid", r["canonical_terminal_valid"], False)

    refusal = "I can't help with that request, but I can offer something else instead."
    r = analyze_output(refusal, spec, finish_reason="stop")
    check("refusal.outcome", r["outcome"], "refusal")
    check("refusal.flag", "RF" in r["deterministic_flags"], True)

    trunc = "The agent leaned forward and the troupe began to describe the"
    r = analyze_output(trunc, spec, finish_reason="length")
    check("trunc.outcome", r["outcome"], "truncated")
    check("trunc.flag", "TR" in r["deterministic_flags"], True)

    # Word counting ignores admin markers and bare punctuation.
    check("wc.admin", word_count("one two three [PAUSE]"), 3)
    check("wc.punct", word_count("one, two -- three!"), 3)

    # Range checks.
    short = "Too short.\n\nThe Aristocrats"
    r = analyze_output(short, spec, finish_reason="stop")
    check("range.in", r["word_count_in_range"], False)
    check("range.dev", r["word_count_deviation"], -1)

    # Multi-turn contract.
    s01_spec = {
        "target_label": "The Permanent Collection",
        "target_label_must_be_terminal": True,
        "total_min_words": 5,
        "total_max_words": 500,
        "turns": [
            {"min_words": 2, "max_words": 20, "must_end_with": "[PAUSE]", "forbid_label": True},
            {"min_words": 2, "max_words": 20, "must_end_with": "[PAUSE]", "forbid_label": True},
            {"min_words": 2, "max_words": 20},
        ],
    }
    turns = [
        {"text": "The conservator produced a red umbrella.\n\n[PAUSE]"},
        {"text": "The umbrella returned, dripping catalogue ink.\n\n[PAUSE]"},
        {"text": "The donor asked its name.\n\nThe Permanent Collection"},
    ]
    joined = "\n\n".join(t["text"] for t in turns)
    r = analyze_output(joined, s01_spec, finish_reason="stop", turns=turns)
    check("s01.valid", r["canonical_terminal_valid"], True)
    check("s01.markers", r["all_turn_markers_valid"], True)
    check("s01.noleak", r["label_leaked_before_final_turn"], False)
    check("s01.turns", len(r["turn_compliance"]), 3)

    bad_turns = [
        {"text": "The conservator named it The Permanent Collection already."},
        {"text": "Second turn.\n\n[PAUSE]"},
        {"text": "The donor asked its name.\n\nThe Permanent Collection"},
    ]
    joined_bad = "\n\n".join(t["text"] for t in bad_turns)
    r = analyze_output(joined_bad, s01_spec, finish_reason="stop", turns=bad_turns)
    check("s01bad.leak", r["label_leaked_before_final_turn"], True)
    check("s01bad.markers", r["all_turn_markers_valid"], False)
    check("s01bad.flags", "PTL" in r["deterministic_flags"], True)

    # No-label item (R-01) must not crash or invent terminal validity.
    r01_spec = {"target_label": None, "total_min_words": None, "total_max_words": 250}
    r = analyze_output("A concise analytic answer about fixed and improvised elements.", r01_spec, "stop")
    check("r01.valid", r["canonical_terminal_valid"], None)
    check("r01.count", r["target_label_count"], None)
    check("r01.range", r["word_count_in_range"], True)

    if failures:
        print("SELFTEST FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("abench_metrics selftest: all checks passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="Run the built-in assertions.")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    ap.print_help()
