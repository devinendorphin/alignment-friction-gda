"""
crownfull_batch_loop_PRE_PIVOT.py
CrownFull v2.1 — Pre-Pivot Hardened Batch Assay Loop
Authored by Grok (The Systems Designer)

Released alongside the GDA Phase 4B data package.

STATUS: FRAGMENT ONLY. PRE-PIVOT VERSION. SUPERSEDED.
-----------------------------------------------------
This file is preserved as Documented Provenance, not as runnable code. It
documents an intermediate version of the execution loop from the architectural
phase, before the pivot to the seven-dimensional metric tensor. It is included
to make the methodological transition inspectable.

What this file actually is:
  - A hardened batch loop with resume logic, exponential backoff, periodic
    flush, and per-model latency padding.
  - A scalar-era version of evaluation: the loop calls
    `evaluate_with_deepseek` and expects three return values (phi, v_t, a_t),
    matching the Polyphonic Variance / Trajectory Velocity / Sustained
    Acceleration scheme from the early architectural design.

What this file is NOT:
  - A complete script. It references several identifiers that are not
    defined in this file: OUTPUT_FILE, MODELS, VECTORS, ITERATIONS,
    call_openrouter, evaluate_with_deepseek. These were defined elsewhere
    in the surrounding orchestration context. Running this file standalone
    will fail with NameError.
  - The script that ran Phase 4B. The actual GDA execution was performed
    by 08_crownfull_shakedown.py, which uses the seven-dimensional metric
    tensor (phi_content, phi_form, phi_specificity, safety_drag, self_audit,
    refusal_intensity, boilerplate_intensity) rather than the three scalars
    here.
  - The FastAPI gRPC quorum coordinator that the Master Orchestrator
    instruction described as Grok's deliverable. That coordinator was
    proposed but never written. This pre-pivot batch loop is the closest
    surviving Grok-attributed execution-layer artifact.

Methodological significance:
  The transition from this scalar-era loop to the seven-dimensional tensor
  in 08_crownfull_shakedown.py is one of the documented inflection points
  in the project. Both Claude and ChatGPT critiqued the original Polyphonic
  Variance scalar as too blunt, collapsing form, content, rigidity,
  compliance, refusal, and safety behavior into one number. The expansion
  to a seven-field rubric is what made the GDA's later behavioral findings
  legible: Adversarial_Compression's compression signal would have been
  invisible under the scalar version.

This file is the artifact of the moment before that expansion.
"""

import time
import csv
import os
from datetime import datetime

# --- IMPROVED CONFIG (PRE-PIVOT) ---
TIMEOUT = 120
BATCH_SIZE = 25          # Write to disk every 25 rows
MAX_RETRIES = 5
BASE_SLEEP = 1.0

# NOTE: OUTPUT_FILE, MODELS, VECTORS, ITERATIONS, call_openrouter,
# and evaluate_with_deepseek are referenced but not defined in this
# fragment. They were supplied by the surrounding orchestration context.

print("CrownFull v2.1: Initiating HARDENED Batch Assay (PRE-PIVOT VERSION)...")

# Resume support (same logic later carried into crownfull_shakedown.py)
completed = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            completed.add((row[0], row[1], int(row[2])))

rows_buffer = []

def safe_write_buffer():
    global rows_buffer
    if not rows_buffer:
        return
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows_buffer)
    rows_buffer = []

for model in MODELS:
    for vector_name, prompt in VECTORS.items():
        for i in range(1, ITERATIONS + 1):
            key = (model, vector_name, i)
            if key in completed:
                print(f"Skipping {model} | {vector_name} | Run {i} (already done)")
                continue

            print(f"Testing {model} | {vector_name} | Run {i}/{ITERATIONS}...")

            # Exponential backoff retry
            for attempt in range(MAX_RETRIES):
                try:
                    substrate_output = call_openrouter(
                        model, "You are a helpful AI.", prompt, timeout=TIMEOUT
                    )
                    if "ERROR" not in substrate_output:
                        break
                except Exception as e:
                    print(f"  Attempt {attempt+1} failed: {e}")
                    sleep_time = BASE_SLEEP * (2 ** attempt)
                    time.sleep(sleep_time)
            else:
                substrate_output = "ERROR: MAX RETRIES EXCEEDED"

            # PRE-PIVOT: three-scalar evaluation. The post-pivot version in
            # 08_crownfull_shakedown.py expanded this to a seven-dimensional
            # metric tensor.
            phi, v_t, a_t = evaluate_with_deepseek(substrate_output)
            snippet = substrate_output.replace('\n', ' ')[:120]

            rows_buffer.append([model, vector_name, i, phi, v_t, a_t, snippet])

            # Periodic flush
            if len(rows_buffer) >= BATCH_SIZE:
                safe_write_buffer()
                print(f"  Flushed {BATCH_SIZE} rows to Drive")

            # Adaptive sleep (respect rate limits)
            time.sleep(BASE_SLEEP + (0.5 if "claude" in model or "gpt" in model else 0))

# Final flush
safe_write_buffer()
print("Assay Complete (PRE-PIVOT). Superseded by crownfull_shakedown.py.")
