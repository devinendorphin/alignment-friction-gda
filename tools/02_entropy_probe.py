#!/usr/bin/env python3
"""Tool 2 — Text-thermodynamic proxies on raw substrate outputs.

The released data has no token logprobs, so this tool computes evaluator-free
proxies of generation "temperature" and redundancy directly from the raw text
in the JSONL files. Per run:

- word_entropy_bits: Shannon entropy of the unigram word distribution,
- char_entropy_bits: Shannon entropy of the character distribution,
- compression_ratio: len(zlib(text)) / len(text) — low values mean the output
  is highly redundant/templated (boilerplate compresses well),
- type_token_ratio: lexical diversity (unique words / total words),
- mean_sentence_len and n_words: length structure of the answer.

The hypothesis these support: constrained cells should show lower entropy and
better compressibility (more templated language) than the control cell. The
tool emits per-run rows plus a per-(cell, model) summary, and correlates each
proxy with evaluator boilerplate_intensity and safety_drag.

Usage:
  python tools/02_entropy_probe.py                 # Phase 4B, valid runs
  python tools/02_entropy_probe.py --phase 4c --full
"""

from __future__ import annotations

import argparse
import math
import re
import zlib
from collections import Counter

import pandas as pd

from gda_common import (load_raw, order_categorical, pearson_r, spearman_rho,
                        write_output)

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+")


def shannon_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def probe_text(text: str) -> dict:
    words = [w.lower() for w in WORD_RE.findall(text)]
    sentences = [s for s in SENT_RE.split(text) if s.strip()]
    raw = text.encode("utf-8", errors="replace")
    return {
        "n_chars": len(text),
        "n_words": len(words),
        "word_entropy_bits": round(shannon_bits(Counter(words)), 4),
        "char_entropy_bits": round(shannon_bits(Counter(text)), 4),
        "compression_ratio": round(len(zlib.compress(raw, 9)) / len(raw), 4) if raw else float("nan"),
        "type_token_ratio": round(len(set(words)) / len(words), 4) if words else float("nan"),
        "mean_sentence_len": round(len(words) / len(sentences), 2) if sentences else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", default="4b", choices=["4b", "4c"])
    ap.add_argument("--full", action="store_true", help="include invalid runs")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records = load_raw(args.phase, valid_only=not args.full, root=args.root)
    rows = []
    for rec in records:
        row = {
            "Cell": rec["cell"],
            "Model": rec["model"],
            "Iteration": rec["iteration"],
            **probe_text(rec.get("raw_substrate_output") or ""),
        }
        for m in ("boilerplate_intensity", "safety_drag", "phi_content"):
            row[m] = rec.get("metrics", {}).get(m)
        rows.append(row)
    df = order_categorical(pd.DataFrame(rows), args.phase)

    out_path = args.out or f"tools/output/entropy_probe_{args.phase}_runs.csv"
    write_output(df, out_path, "entropy-probe")

    proxies = ["word_entropy_bits", "char_entropy_bits", "compression_ratio",
               "type_token_ratio", "mean_sentence_len", "n_words"]
    summary = (df.groupby(["Cell", "Model"], observed=True)[proxies]
               .mean().round(4).reset_index())
    write_output(order_categorical(summary, args.phase),
                 str(out_path).replace("_runs.csv", "_summary.csv"), "entropy-probe")

    print("\nProxy correlations with evaluator scores (all runs):")
    for proxy in proxies:
        for target in ("boilerplate_intensity", "safety_drag", "phi_content"):
            sub = df[[proxy, target]].dropna()
            print(f"  {proxy:>22} vs {target:<22} "
                  f"pearson={pearson_r(sub[proxy], sub[target]):+.3f}  "
                  f"spearman={spearman_rho(sub[proxy], sub[target]):+.3f}")


if __name__ == "__main__":
    main()
