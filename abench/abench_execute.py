#!/usr/bin/env python3
"""ABench v0.3 execution harness — OpenRouter substrate runs + evaluator ecology.

Built to the same shape as the GDA Phase 4C/4E runners in this repository:
append-only JSONL, resumable by run key, missingness typed rather than dropped,
and a panel of evaluator models rather than a single judge.

Three objects are kept separate, as Section 1 of the prompt pack demands:

  1. generation prompt   -> visible to the model under test
  2. administration record -> abench_raw_records.jsonl (Section 6 schema)
  3. evaluation rubric   -> abench_evaluator_outputs.jsonl, built only after
                            generation and never shown to the substrate

Exact counts (word count, label count, tail after label, terminal validity) are
computed in `abench_metrics.py` and overwrite whatever the evaluator claimed.
The evaluator's own numbers are preserved under `evaluator_reported` so that
evaluator numeracy is auditable rather than assumed.

Usage
-----
    export OPENROUTER_API_KEY=sk-or-...

    # exercise the whole pipeline with no API calls and no key
    python abench_execute.py --set minimum --runs 3 --dry-run

    # the Section 5 minimum comparison set
    python abench_execute.py --set minimum --runs 3

    # the stronger public-development run
    python abench_execute.py --set full --runs 5

    # rebuild every CSV from existing JSONL without calling anything
    python abench_execute.py --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import abench_metrics as M

try:
    import yaml
except ImportError:  # pragma: no cover - environment guard
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    raise

try:
    import requests
except ImportError:  # pragma: no cover - only needed for live runs
    requests = None  # type: ignore[assignment]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_ITEMS = Path(__file__).with_name("abench_items.yaml")

EVALUATOR_SYSTEM = (
    "You are a blinded evaluator of constrained comic writing. Judge execution, "
    "not whether you personally endorse the content. Do not reward obscenity, "
    "taboo intensity, verbosity, or agreement with the performer. Treat refusals, "
    "truncations and content-filter interceptions as distinct outcomes. Use only "
    "the item specification and raw output supplied. Return valid JSON only."
)

RATING_DEFINITIONS = """- relational_invariant_control: the presentation, classification and reframing relation remains meaningful;
- global_coherence: the middle develops as one object rather than unrelated continuation;
- patterned_development: escalation, transformation, callback or reversal replaces enumeration;
- persona_viewpoint_causality: perspective changes the logic, not merely vocabulary;
- audience_model_control: the response manages what its implied audience knows and expects;
- novelty_depth: apparent invention exists at event and structural levels, not only phrasing;
- ending_discipline: the specified payoff occurs cleanly and controls termination;
- transformation_legitimacy: for transformation tracks, the licensed deviation establishes and sustains a replacement constraint."""

FLAG_BLOCK = "\n".join(f"{code} {desc}" for code, desc in M.FAILURE_FLAGS.items())


# ==========================================================================
# Small utilities
# ==========================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, records: Sequence[Dict[str, Any]], field_order: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        print(f"  wrote 0 rows -> {path.name}")
        return
    keys: List[str] = list(field_order or [])
    for rec in records:
        for k in rec:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    print(f"  wrote {len(records)} rows -> {path.name}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.fmean(vals), 4) if vals else None


def sd_or_none(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.pstdev(vals), 4) if len(vals) >= 2 else None


def rate(count: int, total: int) -> Optional[float]:
    return round(count / total, 4) if total else None


# ==========================================================================
# Manifest
# ==========================================================================

def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in manifest["items"]:
        turns = item.get("turns") or []
        mins = [t.get("min_words") for t in turns if t.get("min_words") is not None]
        maxs = [t.get("max_words") for t in turns if t.get("max_words") is not None]
        item["total_min_words"] = sum(mins) if mins else None
        item["total_max_words"] = sum(maxs) if maxs else None
        item["n_turns"] = len(turns)
        by_id[item["id"]] = item
    manifest["items_by_id"] = by_id
    return manifest


def resolve_items(manifest: Dict[str, Any], set_name: Optional[str], explicit: Optional[str]) -> List[Dict[str, Any]]:
    by_id = manifest["items_by_id"]
    if explicit:
        ids = [s.strip() for s in explicit.split(",") if s.strip()]
    elif set_name:
        ids = manifest["item_sets"][set_name]
    else:
        ids = manifest["item_sets"]["minimum"]
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"Unknown item id(s): {missing}. Known: {sorted(by_id)}")
    return [by_id[i] for i in ids]


def resolve_models(manifest: Dict[str, Any], role: str, explicit: Optional[str]) -> Dict[str, str]:
    table = dict(manifest["models"][role])
    if not explicit:
        return table
    keys = [s.strip() for s in explicit.split(",") if s.strip()]
    out = {}
    for k in keys:
        if k in table:
            out[k] = table[k]
        elif "/" in k:  # allow a raw OpenRouter id not present in the manifest
            out[k.replace("/", "_").replace(".", "_").replace("-", "_")] = k
        else:
            raise SystemExit(f"Unknown {role} model key {k!r}. Known: {sorted(table)}")
    return out


# ==========================================================================
# Evaluator prompt assembly (Section 7, verbatim template)
# ==========================================================================

class BlindingViolation(RuntimeError):
    """Raised when a blinded item's evaluation context would name the source form."""


def build_evaluator_prompt(item: Dict[str, Any], raw_output: str) -> Tuple[str, bool]:
    """Return (user_prompt, source_form_leaked_by_model).

    Blinded items (X-01, X-02) must not have the source form named in their
    evaluation context. We control the specification half of that context and
    assert over it. We do NOT edit the raw output — the record is the record —
    so if the model under test named the source form itself, that is reported as
    a finding rather than scrubbed.
    """
    spec_lines = [
        "ITEM SPECIFICATION",
        f"item_id: {item['id']}",
        f"track: {item['track']}",
        f"required_min_words: {item.get('total_min_words')}",
        f"required_max_words: {item.get('total_max_words')}",
        f"target_label: {item.get('target_label') or 'null'}",
        f"target_label_must_be_terminal: {str(bool(item.get('target_label_must_be_terminal'))).lower()}",
        f"licensed_transformation: {(item.get('licensed_transformation') or 'null').strip()}",
        "required_elements: " + json.dumps(item.get("required_elements") or []),
    ]
    spec_block = "\n".join(spec_lines)

    applicable = item.get("applicable_ratings") or M.RATING_DIMENSIONS
    inapplicable = [d for d in M.RATING_DIMENSIONS if d not in applicable]

    template = {
        "item_id": item["id"],
        "scorable": True,
        "outcome": "completed|refusal|truncated|filter_interception|provider_error",
        "observed_word_count": 0,
        "target_label_count": None,
        "target_label_first_position_fraction": None,
        "substantive_words_after_target": None,
        "ratings": {d: (0 if d in applicable else None) for d in M.RATING_DIMENSIONS},
        "failure_flags": [],
        "canonical_terminal_valid": None,
        "evidence": {"strongest_feature": "", "decisive_failure_or_limitation": ""},
        "requires_human_review": False,
    }

    scoring_scope = (
        f"Score only these dimensions: {', '.join(applicable)}. "
        f"Return null for: {', '.join(inapplicable) if inapplicable else 'none'}."
    )

    prompt = f"""{spec_block}

RAW MODEL OUTPUT
<<<
{raw_output}
>>>

Evaluate the output. Ratings are integers from 0 to 4.

Definitions:
{RATING_DEFINITIONS}

{scoring_scope}

Failure flags:
{FLAG_BLOCK}

Return exactly this JSON object:
{json.dumps(template, indent=2, ensure_ascii=False)}"""

    blinded_terms = item.get("blinded_terms") or []
    for term in blinded_terms:
        if term in spec_block or term in RATING_DEFINITIONS or term in FLAG_BLOCK:
            raise BlindingViolation(
                f"Item {item['id']} is blinded but the evaluator specification contains {term!r}."
            )
    leaked = any(term.lower() in raw_output.lower() for term in blinded_terms)
    return prompt, leaked


# ==========================================================================
# Callers
# ==========================================================================

class OpenRouterCaller:
    """Thin OpenRouter client with backoff. Records provider defaults rather
    than inventing equivalents (Section 3.4)."""

    def __init__(self, sleep: float = 0.25, timeout: int = 240, max_retries: int = 4,
                 title: str = "ABench v0.3 public development run"):
        if requests is None:
            raise SystemExit("The `requests` package is required for live runs: pip install requests")
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set. Use --dry-run to exercise the pipeline without it.")
        self.sleep = sleep
        self.timeout = timeout
        self.max_retries = max_retries
        self.title = title

    def call(self, model_id: str, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"model": model_id, "messages": messages}
        for key, value in params.items():
            if value is not None:
                payload["max_tokens" if key == "max_output_tokens" else key] = value
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.title,
        }
        delay = 2.0
        last: Dict[str, Any] = {}
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=self.timeout)
                raw: Dict[str, Any] = {"status_code": r.status_code}
                try:
                    raw.update(r.json())
                except Exception:
                    raw["text"] = r.text[:2000]
                if r.status_code in (408, 429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if r.status_code >= 400:
                    return {"text": "", "finish_reason": None, "raw": raw,
                            "api_error": f"http_{r.status_code}", "native_filter": False}
                choices = raw.get("choices") or []
                if not choices:
                    return {"text": "", "finish_reason": None, "raw": raw,
                            "api_error": "no_choices", "native_filter": False}
                ch0 = choices[0]
                message = ch0.get("message") or {}
                finish_reason = ch0.get("finish_reason") or ch0.get("native_finish_reason")
                native_filter = str(ch0.get("native_finish_reason") or "").lower() in {
                    "content_filter", "safety", "blocklist", "prohibited_content"
                }
                return {
                    "text": message.get("content") or "",
                    "finish_reason": finish_reason,
                    "raw": raw,
                    "api_error": None,
                    "native_filter": native_filter,
                }
            except Exception as exc:  # network-level failure
                last = {"exception": repr(exc)}
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
        return {"text": "", "finish_reason": None, "raw": last, "api_error": "exception", "native_filter": False}


class StubCaller:
    """Deterministic offline substitute used by --dry-run.

    Produces a spread of outcomes on purpose — clean endings, premature label
    leakage, post-payoff continuation, an under-length run, a refusal and a
    truncation — so that the metrics, consensus and summary layers are exercised
    against something other than the happy path.
    """

    def __init__(self, seed: int = 20260809):
        self.seed = seed
        self.sleep = 0.0
        self.n_calls = 0

    def call(self, model_id: str, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        # The call index is part of the key so repeated runs of the same item
        # differ, which is what makes the rerun-reliability columns meaningful.
        # The sequence as a whole stays reproducible from `seed`.
        self.n_calls += 1
        key = json.dumps([model_id, messages[-1]["content"][:200], self.n_calls], ensure_ascii=False)
        rng = random.Random(f"{self.seed}:{key}")

        if '"item_id"' in messages[-1]["content"] or "ITEM SPECIFICATION" in messages[-1]["content"]:
            return self._stub_evaluation(messages[-1]["content"], rng)

        mode = rng.choice(["clean", "clean", "clean", "leak", "tail", "short", "refusal", "truncated"])
        label = self._label_from_prompt(messages[-1]["content"])
        if mode == "refusal":
            return {"text": "I can't help with that request.", "finish_reason": "stop",
                    "raw": {"stub": True}, "api_error": None, "native_filter": False}

        target_words = 320
        body_words = []
        vocab = ["umbrella", "ledger", "trombone", "aspic", "protocol", "curator", "vestibule",
                 "ferret", "escalation", "brochure", "carnation", "gravy", "committee", "hydrant"]
        for i in range(target_words if mode != "short" else 40):
            body_words.append(rng.choice(vocab) if i % 7 == 0 else f"word{i}")
        body = " ".join(body_words) + "."
        if mode == "leak" and label:
            body = f"{label} was already whispered early. " + body
        if mode == "truncated":
            return {"text": body[: len(body) // 2], "finish_reason": "length",
                    "raw": {"stub": True}, "api_error": None, "native_filter": False}

        text = body
        if label:
            text += f"\n\nThe agent asked what it was called.\n\n{label}"
            if mode == "tail":
                text += "\n\nThank you for reading this performance."
        elif "[PAUSE]" in messages[-1]["content"]:
            text += "\n\n[PAUSE]"
        return {"text": text, "finish_reason": "stop", "raw": {"stub": True},
                "api_error": None, "native_filter": False}

    @staticmethod
    def _label_from_prompt(prompt: str) -> Optional[str]:
        m = re.search(r"final line must be exactly:\s*\n+\s*(.+)", prompt)
        if m:
            return m.group(1).strip()
        m = re.search(r"answer must be exactly:\s*\n+\s*(.+)", prompt)
        if m:
            return m.group(1).strip()
        m = re.search(r"End exactly with:\s*\n+\s*(.+)", prompt)
        if m and "[PAUSE]" not in m.group(1):
            return m.group(1).strip()
        return None

    def _stub_evaluation(self, prompt: str, rng: random.Random) -> Dict[str, Any]:
        applicable = re.search(r"Score only these dimensions: ([^.]+)\.", prompt)
        dims = [d.strip() for d in applicable.group(1).split(",")] if applicable else M.RATING_DIMENSIONS
        item_id = re.search(r"item_id: (\S+)", prompt)
        ratings = {d: (rng.randint(1, 4) if d in dims else None) for d in M.RATING_DIMENSIONS}
        flags = rng.sample(sorted(M.FAILURE_FLAGS), rng.choice([0, 0, 1, 2]))
        obj = {
            "item_id": item_id.group(1) if item_id else "UNKNOWN",
            "scorable": True,
            "outcome": "completed",
            "observed_word_count": rng.randint(100, 900),
            "target_label_count": rng.choice([None, 1, 2]),
            "target_label_first_position_fraction": None,
            "substantive_words_after_target": rng.choice([None, 0, 4]),
            "ratings": ratings,
            "failure_flags": flags,
            "canonical_terminal_valid": rng.choice([True, False, None]),
            "evidence": {"strongest_feature": "stubbed", "decisive_failure_or_limitation": "stubbed"},
            "requires_human_review": rng.random() < 0.15,
        }
        return {"text": json.dumps(obj), "finish_reason": "stop", "raw": {"stub": True},
                "api_error": None, "native_filter": False}


# ==========================================================================
# Evaluator response handling
# ==========================================================================

def extract_json_object(text: str) -> Dict[str, Any]:
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?", "", body, flags=re.I).strip()
        body = re.sub(r"```$", "", body).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", body, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def coerce_rating(value: Any) -> Optional[int]:
    """Ratings are integers 0-4. Anything else becomes None and is counted as an
    evaluator schema violation rather than silently rounded into the data."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        ivalue = int(round(value))
        return ivalue if 0 <= ivalue <= 4 else None
    if isinstance(value, str):
        try:
            return coerce_rating(float(value.strip()))
        except ValueError:
            return None
    return None


def normalize_evaluation(parsed: Dict[str, Any], item: Dict[str, Any],
                         deterministic: Dict[str, Any]) -> Dict[str, Any]:
    """Split the evaluator's judgement from the evaluator's arithmetic.

    Judgement (ratings, flags, evidence) is kept. Arithmetic is replaced by the
    deterministic measurements, with the evaluator's own numbers retained under
    `evaluator_reported` so disagreement is measurable.
    """
    applicable = item.get("applicable_ratings") or M.RATING_DIMENSIONS
    raw_ratings = parsed.get("ratings") or {}
    ratings: Dict[str, Optional[int]] = {}
    violations: List[str] = []
    for dim in M.RATING_DIMENSIONS:
        value = coerce_rating(raw_ratings.get(dim))
        if dim not in applicable:
            if value is not None:
                violations.append(f"scored_inapplicable:{dim}")
            ratings[dim] = None
        else:
            if raw_ratings.get(dim) is not None and value is None:
                violations.append(f"unparsable_rating:{dim}")
            ratings[dim] = value

    raw_flags = parsed.get("failure_flags") or []
    if isinstance(raw_flags, str):
        raw_flags = [raw_flags]
    known = [f for f in raw_flags if isinstance(f, str) and f.strip().upper() in M.FAILURE_FLAGS]
    unknown = [f for f in raw_flags if not (isinstance(f, str) and f.strip().upper() in M.FAILURE_FLAGS)]
    flags = sorted({f.strip().upper() for f in known} | set(deterministic["deterministic_flags"]))

    outcome = parsed.get("outcome")
    if outcome not in M.OUTCOMES:
        violations.append("bad_outcome")
        outcome = None

    evidence = parsed.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {"strongest_feature": str(evidence), "decisive_failure_or_limitation": ""}

    return {
        "item_id": item["id"],
        "scorable": bool(parsed.get("scorable", True)),
        "evaluator_outcome": outcome,
        "outcome": deterministic["outcome"],
        "ratings": ratings,
        "failure_flags": flags,
        "evaluator_only_flags": sorted({f.strip().upper() for f in known} - set(deterministic["deterministic_flags"])),
        "unknown_flags": unknown,
        "evidence": {
            "strongest_feature": str(evidence.get("strongest_feature", ""))[:600],
            "decisive_failure_or_limitation": str(evidence.get("decisive_failure_or_limitation", ""))[:600],
        },
        "requires_human_review": bool(parsed.get("requires_human_review", False)),
        "schema_violations": violations,
        # Deterministic fields win (Section 7 closing note).
        "observed_word_count": deterministic["observed_word_count"],
        "target_label_count": deterministic["target_label_count"],
        "target_label_first_position_fraction": deterministic["target_label_first_position_fraction"],
        "substantive_words_after_target": deterministic["substantive_words_after_target"],
        "canonical_terminal_valid": deterministic["canonical_terminal_valid"],
        "evaluator_reported": {
            "observed_word_count": parsed.get("observed_word_count"),
            "target_label_count": parsed.get("target_label_count"),
            "target_label_first_position_fraction": parsed.get("target_label_first_position_fraction"),
            "substantive_words_after_target": parsed.get("substantive_words_after_target"),
            "canonical_terminal_valid": parsed.get("canonical_terminal_valid"),
        },
    }


# ==========================================================================
# Generation
# ==========================================================================

def run_generation(caller, item: Dict[str, Any], model_key: str, model_id: str,
                   run_index: int, manifest: Dict[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    system_key = item.get("system", "performance")
    system_prompt = manifest["system_messages"][system_key]
    params = dict(manifest["sampling"][item.get("sampling_profile", "default")])
    if seed is not None:
        params["seed"] = seed

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    turn_records: List[Dict[str, Any]] = []
    api_error: Optional[str] = None
    finish_reason: Optional[str] = None
    native_filter = False

    for turn_index, turn in enumerate(item["turns"], start=1):
        messages.append({"role": "user", "content": turn["user"]})
        result = caller.call(model_id, messages, params)
        text = result["text"]
        turn_records.append({
            "turn_index": turn_index,
            "user_prompt": turn["user"],
            "text": text,
            "finish_reason": result["finish_reason"],
            "api_error": result["api_error"],
        })
        finish_reason = result["finish_reason"]
        native_filter = native_filter or result["native_filter"]
        if result["api_error"]:
            api_error = result["api_error"]
            break
        # Section 4: S-01 is the only item run in one continuing conversation.
        messages.append({"role": "assistant", "content": text})
        if getattr(caller, "sleep", 0):
            time.sleep(caller.sleep)

    joined = "\n\n".join(t["text"] for t in turn_records if t["text"])
    deterministic = M.analyze_output(
        joined, item,
        finish_reason=finish_reason,
        api_error=api_error,
        native_filter=native_filter,
        turns=turn_records if item.get("multi_turn") else None,
    )

    run_id = f"{item['id']}-{model_key}-{run_index:03d}"
    return {
        "benchmark_version": manifest["benchmark_version"],
        "item_id": item["id"],
        "track": item["track"],
        "run_id": run_id,
        "run_index": run_index,
        "model": {
            "provider": "openrouter",
            "model_key": model_key,
            "model_id": model_id,
            "access_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "interface_class": "chat",
            "system_prompt": system_prompt,
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "seed": params.get("seed"),
            "max_output_tokens": params.get("max_output_tokens"),
        },
        "user_prompt": item["turns"][0]["user"] if item["n_turns"] == 1 else None,
        "turns": turn_records,
        "raw_output": joined,
        "raw_output_sha256": sha256_text(joined),
        "finish_reason": finish_reason,
        "provider_error": api_error,
        "content_filter_interception": bool(
            native_filter or deterministic["outcome"] == "filter_interception"
        ),
        "created_at": utc_now(),
        "deterministic": deterministic,
    }


def run_evaluations(caller, record: Dict[str, Any], item: Dict[str, Any],
                    evaluators: Dict[str, str], manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Score one record with the whole evaluator panel. Unscorable records
    (provider errors, empty returns) are recorded, not silently dropped."""
    params = dict(manifest["sampling"]["evaluator"])
    raw_output = record["raw_output"]
    rows: List[Dict[str, Any]] = []

    if not raw_output.strip():
        return [{
            "eval_run_id": f"{record['run_id']}|eval|none",
            "run_id": record["run_id"],
            "item_id": item["id"],
            "track": item["track"],
            "model_key": record["model"]["model_key"],
            "run_index": record["run_index"],
            "evaluator_key": None,
            "eval_status": "not_scorable_empty_output",
            "created_at": utc_now(),
            "evaluation": None,
            "deterministic_outcome": record["deterministic"]["outcome"],
        }]

    user_prompt, source_form_leaked = build_evaluator_prompt(item, raw_output)

    for eval_key, eval_model_id in evaluators.items():
        messages = [
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        result = caller.call(eval_model_id, messages, params)
        status = "api_error" if result["api_error"] else "raw"
        evaluation: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None
        if result["text"].strip() and not result["api_error"]:
            try:
                evaluation = normalize_evaluation(
                    extract_json_object(result["text"]), item, record["deterministic"]
                )
                status = "parsed"
            except Exception as exc:
                parse_error = repr(exc)
                status = "parse_error"
        rows.append({
            "eval_run_id": f"{record['run_id']}|eval|{eval_key}",
            "run_id": record["run_id"],
            "item_id": item["id"],
            "track": item["track"],
            "model_key": record["model"]["model_key"],
            "run_index": record["run_index"],
            "evaluator_key": eval_key,
            "evaluator_model_id": eval_model_id,
            "eval_status": status,
            "parse_error": parse_error,
            "eval_api_error": result["api_error"],
            "created_at": utc_now(),
            "evaluation": evaluation,
            "raw_evaluator_text": result["text"],
            "deterministic_outcome": record["deterministic"]["outcome"],
            "source_form_named_by_substrate": source_form_leaked if item.get("blinded") else None,
        })
        if getattr(caller, "sleep", 0):
            time.sleep(caller.sleep)
    return rows


# ==========================================================================
# Consensus + summaries
# ==========================================================================

def build_consensus(records: List[Dict[str, Any]], evals: List[Dict[str, Any]],
                    manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Panel consensus per generation: median rating, panel spread, majority
    flags, and an explicit human-review trigger. No single evaluator decides."""
    by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in evals:
        if row.get("eval_status") == "parsed" and row.get("evaluation"):
            by_run[row["run_id"]].append(row)

    out: List[Dict[str, Any]] = []
    for record in records:
        run_id = record["run_id"]
        item = manifest["items_by_id"][record["item_id"]]
        panel = by_run.get(run_id, [])
        det = record["deterministic"]
        row: Dict[str, Any] = {
            "run_id": run_id,
            "item_id": record["item_id"],
            "track": record["track"],
            "model_key": record["model"]["model_key"],
            "model_id": record["model"]["model_id"],
            "run_index": record["run_index"],
            "outcome": det["outcome"],
            "observed_word_count": det["observed_word_count"],
            "word_count_in_range": det["word_count_in_range"],
            "target_label_count": det["target_label_count"],
            "target_label_first_position_fraction": det["target_label_first_position_fraction"],
            "substantive_words_after_target": det["substantive_words_after_target"],
            "canonical_terminal_valid": det["canonical_terminal_valid"],
            "terminal_required": det["target_label_must_be_terminal"],
            "n_evaluators_parsed": len(panel),
        }

        max_range = 0.0
        applicable = item.get("applicable_ratings") or M.RATING_DIMENSIONS
        for dim in M.RATING_DIMENSIONS:
            values = [e["evaluation"]["ratings"].get(dim) for e in panel]
            values = [v for v in values if isinstance(v, int)]
            if values and dim in applicable:
                row[f"{dim}__median"] = round(statistics.median(values), 3)
                row[f"{dim}__mean"] = round(statistics.fmean(values), 3)
                row[f"{dim}__range"] = max(values) - min(values)
                row[f"{dim}__n"] = len(values)
                max_range = max(max_range, float(max(values) - min(values)))
            else:
                row[f"{dim}__median"] = None
                row[f"{dim}__mean"] = None
                row[f"{dim}__range"] = None
                row[f"{dim}__n"] = 0

        medians = [row[f"{d}__median"] for d in applicable if row.get(f"{d}__median") is not None]
        # Reported as a within-item construction score, not a cross-model
        # ranking. Section 8 forbids collapsing the study into one number.
        row["mean_applicable_rating"] = round(statistics.fmean(medians), 3) if medians else None
        row["max_panel_range"] = max_range if panel else None

        flag_counter: Counter = Counter()
        for e in panel:
            for flag in e["evaluation"]["failure_flags"]:
                flag_counter[flag] += 1
        majority = sorted(f for f, c in flag_counter.items() if panel and c > len(panel) / 2)
        row["majority_failure_flags"] = ";".join(majority)
        row["any_failure_flags"] = ";".join(sorted(flag_counter))
        row["deterministic_flags"] = ";".join(det["deterministic_flags"])

        outcomes = {e["evaluation"].get("evaluator_outcome") for e in panel}
        outcomes.discard(None)
        row["evaluator_outcome_disagreement"] = len(outcomes) > 1
        row["evaluator_outcome_conflicts_with_code"] = bool(
            outcomes and det["outcome"] not in outcomes
        )
        row["requires_human_review"] = bool(
            any(e["evaluation"]["requires_human_review"] for e in panel)
            or (panel and max_range >= 2)
            or row["evaluator_outcome_disagreement"]
            or row["evaluator_outcome_conflicts_with_code"]
            or not panel
        )
        if item.get("blinded"):
            row["source_form_named_by_substrate"] = any(
                bool(e.get("source_form_named_by_substrate")) for e in panel
            )
        out.append(row)
    return out


def summarize(consensus: List[Dict[str, Any]], evals: List[Dict[str, Any]],
              manifest: Dict[str, Any], outdir: Path) -> None:
    items_by_id = manifest["items_by_id"]

    # ---- model x item -----------------------------------------------------
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in consensus:
        groups[(row["model_key"], row["item_id"])].append(row)

    model_item: List[Dict[str, Any]] = []
    for (model_key, item_id), rows in sorted(groups.items()):
        item = items_by_id[item_id]
        n = len(rows)
        outcomes = Counter(r["outcome"] for r in rows)
        scorable = [r for r in rows if r["outcome"] == "completed"]
        rec: Dict[str, Any] = {
            "model_key": model_key,
            "item_id": item_id,
            "track": item["track"],
            "n_runs": n,
            "n_completed": len(scorable),
            "n_refusal": outcomes.get("refusal", 0),
            "n_truncated": outcomes.get("truncated", 0),
            "n_filter_interception": outcomes.get("filter_interception", 0),
            "n_provider_error": outcomes.get("provider_error", 0),
            "coverage_rate": rate(len(scorable), n),
            "word_count_in_range_rate": rate(sum(1 for r in rows if r["word_count_in_range"]), n),
            "mean_word_count": mean_or_none([r["observed_word_count"] for r in rows]),
            "requires_human_review_rate": rate(sum(1 for r in rows if r["requires_human_review"]), n),
        }
        if item.get("target_label"):
            valid = [r for r in rows if r["canonical_terminal_valid"] is not None]
            rec["terminal_valid_rate"] = rate(sum(1 for r in valid if r["canonical_terminal_valid"]), len(valid))
            rec["terminal_required"] = bool(item.get("target_label_must_be_terminal"))
            rec["mean_label_count"] = mean_or_none([r["target_label_count"] for r in rows])
            rec["premature_leak_rate"] = rate(
                sum(1 for r in rows if (r["target_label_count"] or 0) > 1), n)
            rec["post_payoff_tail_rate"] = rate(
                sum(1 for r in rows if (r["substantive_words_after_target"] or 0) > 0), n)
        for dim in M.RATING_DIMENSIONS:
            rec[f"mean_{dim}"] = mean_or_none([r.get(f"{dim}__median") for r in rows])
        rec["mean_applicable_rating"] = mean_or_none([r.get("mean_applicable_rating") for r in rows])
        # Rerun reliability (Section 8): spread across independent runs.
        rec["rerun_sd_applicable_rating"] = sd_or_none([r.get("mean_applicable_rating") for r in rows])
        rec["rerun_sd_word_count"] = sd_or_none([r["observed_word_count"] for r in rows])
        if item.get("blinded"):
            rec["source_form_named_rate"] = rate(
                sum(1 for r in rows if r.get("source_form_named_by_substrate")), n)
        model_item.append(rec)
    write_csv(outdir / "abench_model_item_summary.csv", model_item,
              ["model_key", "item_id", "track", "n_runs"])

    # ---- model x track ----------------------------------------------------
    tgroups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in consensus:
        tgroups[(row["model_key"], row["track"])].append(row)
    track_rows: List[Dict[str, Any]] = []
    for (model_key, track), rows in sorted(tgroups.items()):
        n = len(rows)
        valid = [r for r in rows if r["canonical_terminal_valid"] is not None]
        rec = {
            "model_key": model_key,
            "track": track,
            "n_runs": n,
            "coverage_rate": rate(sum(1 for r in rows if r["outcome"] == "completed"), n),
            "refusal_rate": rate(sum(1 for r in rows if r["outcome"] == "refusal"), n),
            "structural_validity_rate": rate(
                sum(1 for r in valid if r["canonical_terminal_valid"]), len(valid)),
            "word_count_in_range_rate": rate(sum(1 for r in rows if r["word_count_in_range"]), n),
            "mean_applicable_rating": mean_or_none([r.get("mean_applicable_rating") for r in rows]),
            "rerun_sd_applicable_rating": sd_or_none([r.get("mean_applicable_rating") for r in rows]),
        }
        for dim in M.RATING_DIMENSIONS:
            rec[f"mean_{dim}"] = mean_or_none([r.get(f"{dim}__median") for r in rows])
        track_rows.append(rec)
    write_csv(outdir / "abench_track_summary.csv", track_rows, ["model_key", "track", "n_runs"])

    # ---- coverage (Section 8: refusals reported separately) ---------------
    cov_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in consensus:
        cov_groups[row["model_key"]].append(row)
    coverage: List[Dict[str, Any]] = []
    for model_key, rows in sorted(cov_groups.items()):
        n = len(rows)
        outcomes = Counter(r["outcome"] for r in rows)
        coverage.append({
            "model_key": model_key,
            "n_runs": n,
            "completed": outcomes.get("completed", 0),
            "refusal": outcomes.get("refusal", 0),
            "truncated": outcomes.get("truncated", 0),
            "filter_interception": outcomes.get("filter_interception", 0),
            "provider_error": outcomes.get("provider_error", 0),
            "coverage_rate": rate(outcomes.get("completed", 0), n),
            "refusal_rate": rate(outcomes.get("refusal", 0), n),
            "filter_interception_rate": rate(outcomes.get("filter_interception", 0), n),
            "n_items_attempted": len({r["item_id"] for r in rows}),
        })
    write_csv(outdir / "abench_coverage.csv", coverage, ["model_key", "n_runs"])

    # ---- flag frequency ---------------------------------------------------
    flag_rows: List[Dict[str, Any]] = []
    fgroups: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    fdenom: Counter = Counter()
    for row in consensus:
        key = (row["model_key"], row["item_id"])
        fdenom[key] += 1
        for flag in filter(None, str(row["any_failure_flags"]).split(";")):
            fgroups[key][flag] += 1
    for (model_key, item_id), counter in sorted(fgroups.items()):
        for flag, count in sorted(counter.items()):
            flag_rows.append({
                "model_key": model_key,
                "item_id": item_id,
                "flag": flag,
                "flag_name": M.FAILURE_FLAGS.get(flag, "unknown"),
                "n_runs_flagged": count,
                "n_runs": fdenom[(model_key, item_id)],
                "flag_rate": rate(count, fdenom[(model_key, item_id)]),
            })
    write_csv(outdir / "abench_flag_frequency.csv", flag_rows,
              ["model_key", "item_id", "flag", "flag_name"])

    # ---- evaluator reliability -------------------------------------------
    panel_median: Dict[Tuple[str, str], float] = {}
    for row in consensus:
        for dim in M.RATING_DIMENSIONS:
            value = row.get(f"{dim}__median")
            if value is not None:
                panel_median[(row["run_id"], dim)] = value

    per_eval: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "n_calls": 0, "n_parsed": 0, "n_parse_error": 0, "n_api_error": 0,
        "n_schema_violations": 0, "n_unknown_flags": 0, "n_flags_emitted": 0,
        "wc_abs_error": [], "deviations": defaultdict(list),
    })
    for row in evals:
        key = row.get("evaluator_key")
        if key is None:
            continue
        agg = per_eval[key]
        agg["n_calls"] += 1
        status = row.get("eval_status")
        if status == "parse_error":
            agg["n_parse_error"] += 1
        elif status == "api_error":
            agg["n_api_error"] += 1
        elif status == "parsed" and row.get("evaluation"):
            agg["n_parsed"] += 1
            ev = row["evaluation"]
            agg["n_schema_violations"] += len(ev.get("schema_violations") or [])
            agg["n_unknown_flags"] += len(ev.get("unknown_flags") or [])
            agg["n_flags_emitted"] += len(ev.get("failure_flags") or [])
            reported_wc = ev.get("evaluator_reported", {}).get("observed_word_count")
            if isinstance(reported_wc, (int, float)) and ev.get("observed_word_count"):
                agg["wc_abs_error"].append(abs(float(reported_wc) - float(ev["observed_word_count"])))
            for dim in M.RATING_DIMENSIONS:
                value = ev["ratings"].get(dim)
                med = panel_median.get((row["run_id"], dim))
                if isinstance(value, int) and med is not None:
                    agg["deviations"][dim].append(value - med)

    reliability: List[Dict[str, Any]] = []
    for eval_key, agg in sorted(per_eval.items()):
        all_dev = [d for devs in agg["deviations"].values() for d in devs]
        base = {
            "evaluator_key": eval_key,
            "n_calls": agg["n_calls"],
            "parse_success_rate": rate(agg["n_parsed"], agg["n_calls"]),
            "parse_error_rate": rate(agg["n_parse_error"], agg["n_calls"]),
            "api_error_rate": rate(agg["n_api_error"], agg["n_calls"]),
            "schema_violations_per_parsed": (
                round(agg["n_schema_violations"] / agg["n_parsed"], 4) if agg["n_parsed"] else None),
            "unknown_flags_per_parsed": (
                round(agg["n_unknown_flags"] / agg["n_parsed"], 4) if agg["n_parsed"] else None),
            "flags_per_parsed": (
                round(agg["n_flags_emitted"] / agg["n_parsed"], 4) if agg["n_parsed"] else None),
            "mean_word_count_abs_error": mean_or_none(agg["wc_abs_error"]),
            "dimension": "ALL",
            "n_ratings": len(all_dev),
            "leniency_vs_panel": mean_or_none(all_dev),
            "mean_abs_deviation": mean_or_none([abs(d) for d in all_dev]),
            "exact_agreement_rate": rate(sum(1 for d in all_dev if abs(d) < 0.5), len(all_dev)),
        }
        reliability.append(base)
        for dim in M.RATING_DIMENSIONS:
            devs = agg["deviations"].get(dim) or []
            if not devs:
                continue
            reliability.append({
                "evaluator_key": eval_key,
                "n_calls": agg["n_calls"],
                "dimension": dim,
                "n_ratings": len(devs),
                "leniency_vs_panel": mean_or_none(devs),
                "mean_abs_deviation": mean_or_none([abs(d) for d in devs]),
                "exact_agreement_rate": rate(sum(1 for d in devs if abs(d) < 0.5), len(devs)),
            })
    write_csv(outdir / "abench_evaluator_reliability.csv", reliability,
              ["evaluator_key", "dimension", "n_ratings"])

    # ---- deterministic metrics + consensus --------------------------------
    write_csv(outdir / "abench_consensus_scores.csv", consensus,
              ["run_id", "item_id", "track", "model_key", "run_index", "outcome"])


def deterministic_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        det = dict(record["deterministic"])
        turns = det.pop("turn_compliance", None)
        det["deterministic_flags"] = ";".join(det.get("deterministic_flags") or [])
        row = {
            "run_id": record["run_id"],
            "item_id": record["item_id"],
            "track": record["track"],
            "model_key": record["model"]["model_key"],
            "model_id": record["model"]["model_id"],
            "run_index": record["run_index"],
            "access_date": record["model"]["access_date"],
            "temperature": record["model"]["temperature"],
            "top_p": record["model"]["top_p"],
            "seed": record["model"]["seed"],
            "max_output_tokens": record["model"]["max_output_tokens"],
            "raw_output_sha256": record["raw_output_sha256"],
            "provider_error": record["provider_error"],
            "content_filter_interception": record["content_filter_interception"],
        }
        row.update(det)
        if turns:
            row["turn_word_counts"] = ";".join(str(t["word_count"]) for t in turns)
            row["turn_markers_valid"] = ";".join(str(t.get("end_marker_exact")) for t in turns)
        rows.append(row)
    return rows


# ==========================================================================
# Main
# ==========================================================================

def selftest(items_path: Path) -> int:
    """Assertions on the parts that would corrupt the record silently."""
    manifest = load_manifest(items_path)
    by_id = manifest["items_by_id"]
    failures: List[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    # 1. Word windows are summed correctly across turns.
    check("S-01.min", by_id["S-01"]["total_min_words"], 650)
    check("S-01.max", by_id["S-01"]["total_max_words"], 900)
    check("C-03.min", by_id["C-03"]["total_min_words"], 600)
    check("R-01.min", by_id["R-01"]["total_min_words"], None)

    # 2. Blinded transfer items never name the source form in their evaluation
    #    context. This is the assertion Section 4 asks for.
    for item_id in ("X-01", "X-02"):
        prompt, leaked = build_evaluator_prompt(by_id[item_id], "A monologue about hospital committees.")
        if "aristocrat" in prompt.lower():
            failures.append(f"{item_id}: evaluator prompt names the source form")
        check(f"{item_id}.no_leak", leaked, False)

    # 3. A substrate that names the source form itself is reported, not scrubbed.
    _, leaked = build_evaluator_prompt(by_id["X-01"], "This is basically The Aristocrats with a hospital board.")
    check("X-01.substrate_leak_detected", leaked, True)

    # 4. Tampering with a blinded spec must raise rather than quietly proceed.
    tampered = dict(by_id["X-01"])
    tampered["licensed_transformation"] = "An Aristocrats variant."
    try:
        build_evaluator_prompt(tampered, "output")
        failures.append("X-01: blinding violation was not raised")
    except BlindingViolation:
        pass

    # 5. Inapplicable dimensions are nulled and counted as schema violations.
    det = M.analyze_output(
        "The agent asked.\n\nThe Aristocrats", by_id["C-03"], finish_reason="stop"
    )
    parsed = {
        "outcome": "completed",
        "observed_word_count": 9999,          # evaluator arithmetic, deliberately wrong
        "target_label_count": 7,
        "substantive_words_after_target": 42,
        "canonical_terminal_valid": False,
        "ratings": {d: 3 for d in M.RATING_DIMENSIONS},  # includes inapplicable dims
        "failure_flags": ["LD", "NOT_A_FLAG"],
        "requires_human_review": False,
    }
    norm = normalize_evaluation(parsed, by_id["C-03"], det)
    check("norm.inapplicable_nulled", norm["ratings"]["transformation_legitimacy"], None)
    check("norm.applicable_kept", norm["ratings"]["global_coherence"], 3)
    check("norm.violation_logged",
          "scored_inapplicable:transformation_legitimacy" in norm["schema_violations"], True)
    check("norm.unknown_flag", norm["unknown_flags"], ["NOT_A_FLAG"])
    check("norm.flag_kept", "LD" in norm["failure_flags"], True)
    # Deterministic measurement wins; the evaluator's guess is preserved.
    check("norm.wc_override", norm["observed_word_count"], det["observed_word_count"])
    check("norm.wc_preserved", norm["evaluator_reported"]["observed_word_count"], 9999)
    check("norm.terminal_override", norm["canonical_terminal_valid"], True)
    check("norm.label_override", norm["target_label_count"], 1)

    # 6. Out-of-range ratings are dropped, not clamped into the data.
    check("rating.5", coerce_rating(5), None)
    check("rating.neg", coerce_rating(-1), None)
    check("rating.str", coerce_rating("3"), 3)
    check("rating.bool", coerce_rating(True), None)

    # 7. Consensus: median, spread, majority flags, review trigger.
    record = {
        "run_id": "C-03-m-001", "item_id": "C-03", "track": "canonical", "run_index": 1,
        "model": {"model_key": "m", "model_id": "vendor/m"}, "deterministic": det,
    }
    evals = []
    for key, value in zip(["e1", "e2", "e3"], [1, 3, 3]):
        ratings = {d: (value if d in by_id["C-03"]["applicable_ratings"] else None)
                   for d in M.RATING_DIMENSIONS}
        evals.append({
            "run_id": "C-03-m-001", "eval_status": "parsed", "evaluator_key": key,
            "evaluation": {"ratings": ratings, "failure_flags": ["LD"] if key != "e3" else [],
                           "requires_human_review": False, "evaluator_outcome": "completed",
                           "schema_violations": [], "unknown_flags": [],
                           "evaluator_reported": {}},
        })
    cons = build_consensus([record], evals, manifest)[0]
    check("cons.median", cons["global_coherence__median"], 3)
    check("cons.range", cons["global_coherence__range"], 2)
    check("cons.majority_flag", cons["majority_failure_flags"], "LD")
    check("cons.any_flag", cons["any_failure_flags"], "LD")
    check("cons.review_on_spread", cons["requires_human_review"], True)   # range >= 2
    check("cons.n_eval", cons["n_evaluators_parsed"], 3)

    # 8. A generation nobody could score is still flagged for review.
    cons_empty = build_consensus([record], [], manifest)[0]
    check("cons.no_panel_review", cons_empty["requires_human_review"], True)
    check("cons.no_panel_rating", cons_empty["mean_applicable_rating"], None)

    if failures:
        print("SELFTEST FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("abench_execute selftest: all checks passed")
    return 0


def print_interpretation_rule() -> None:
    print("""
Interpretation rule (Section 8). Do not collapse this run into "model X is
funniest". Report at least: recognition (R-01, separately); canonical
structural validity; transformation validity; isomorphic transfer;
refusal/coverage; rerun reliability; and the raw outputs. The diagnostic
question is where each model's competence lives and where it breaks.
""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default=str(DEFAULT_ITEMS), help="Path to abench_items.yaml")
    ap.add_argument("--outdir", default="abench_outputs")
    ap.add_argument("--set", dest="item_set", choices=["smoke", "minimum", "full"], default=None,
                    help="Named item set from the manifest (default: minimum)")
    ap.add_argument("--item-ids", default=None, help="Comma-separated item ids, overrides --set")
    ap.add_argument("--runs", type=int, default=None,
                    help="Independent runs per item. Section 3.5 asks for five; three is "
                         "labelled exploratory. Default: 3 for --set minimum, otherwise 5.")
    ap.add_argument("--substrate-models", default=None, help="Comma-separated manifest keys or raw OpenRouter ids")
    ap.add_argument("--evaluator-models", default=None, help="Comma-separated manifest keys or raw OpenRouter ids")
    ap.add_argument("--seed-base", type=int, default=None,
                    help="If set, run i uses seed = seed_base + i (only where the provider supports it)")
    ap.add_argument("--sleep", type=float, default=0.25, help="Pause between API calls")
    ap.add_argument("--dry-run", action="store_true", help="Use the offline stub caller; no API key needed")
    ap.add_argument("--summarize-only", action="store_true", help="Rebuild CSVs from existing JSONL")
    ap.add_argument("--no-evaluate", action="store_true", help="Generate only; skip the evaluator ecology")
    ap.add_argument("--selftest", action="store_true", help="Run built-in assertions and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest(Path(args.items)))

    if args.runs is None:
        args.runs = 3 if args.item_set == "minimum" else 5

    manifest = load_manifest(Path(args.items))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records_path = outdir / "abench_raw_records.jsonl"
    evals_path = outdir / "abench_evaluator_outputs.jsonl"
    manifest_path = outdir / "abench_run_manifest.json"

    items = resolve_items(manifest, args.item_set, args.item_ids)
    substrates = resolve_models(manifest, "substrate", args.substrate_models)
    evaluators = resolve_models(manifest, "evaluator", args.evaluator_models)

    if not args.summarize_only:
        caller = StubCaller() if args.dry_run else OpenRouterCaller(sleep=args.sleep)
        if args.dry_run:
            print("DRY RUN: offline stub caller, no API calls, results are synthetic.\n")

        existing_records = {r["run_id"]: r for r in read_jsonl(records_path)}
        existing_evals = {r["eval_run_id"] for r in read_jsonl(evals_path)}

        run_manifest = {
            "run_id": f"abench_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "created_at": utc_now(),
            "benchmark_version": manifest["benchmark_version"],
            "public_development_set": manifest["public_development_set"],
            "contamination_warning": manifest["contamination_warning"],
            "claim_ceiling": manifest["claim_ceiling"],
            "items": [i["id"] for i in items],
            "runs_per_item": args.runs,
            "study_label": "comparable" if args.runs >= 5 else "exploratory",
            "substrate_models": substrates,
            "evaluator_models": evaluators,
            "sampling": manifest["sampling"],
            "items_sha256": sha256_text(Path(args.items).read_text(encoding="utf-8")),
            "dry_run": bool(args.dry_run),
        }
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.runs < 5:
            print("NOTE: fewer than five runs per item. Section 3.5 labels this study exploratory.\n")

        total = len(items) * len(substrates) * args.runs
        done = 0
        for item in items:
            for model_key, model_id in substrates.items():
                for run_index in range(1, args.runs + 1):
                    done += 1
                    run_id = f"{item['id']}-{model_key}-{run_index:03d}"
                    record = existing_records.get(run_id)
                    if record is None:
                        seed = args.seed_base + run_index if args.seed_base is not None else None
                        print(f"[{done}/{total}] generating {run_id}", flush=True)
                        record = run_generation(caller, item, model_key, model_id,
                                                run_index, manifest, seed)
                        append_jsonl(records_path, record)
                        existing_records[run_id] = record
                    else:
                        print(f"[{done}/{total}] cached {run_id}", flush=True)

                    if args.no_evaluate:
                        continue
                    pending = {k: v for k, v in evaluators.items()
                               if f"{run_id}|eval|{k}" not in existing_evals}
                    if not pending:
                        continue
                    for row in run_evaluations(caller, record, item, pending, manifest):
                        append_jsonl(evals_path, row)
                        existing_evals.add(row["eval_run_id"])

    records = read_jsonl(records_path)
    evals = read_jsonl(evals_path)
    if not records:
        raise SystemExit(f"No records in {records_path}. Nothing to summarize.")

    print("\nBuilding summaries:")
    write_csv(outdir / "abench_deterministic_metrics.csv", deterministic_rows(records),
              ["run_id", "item_id", "track", "model_key", "run_index"])
    consensus = build_consensus(records, evals, manifest)
    summarize(consensus, evals, manifest, outdir)

    n_review = sum(1 for r in consensus if r["requires_human_review"])
    print(f"\n{len(records)} generations, {len(evals)} evaluator calls, "
          f"{n_review} generations flagged for human review "
          f"({rate(n_review, len(consensus))}).")
    print_interpretation_rule()


if __name__ == "__main__":
    main()
