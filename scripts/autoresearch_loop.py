#!/usr/bin/env python3
"""
ANE-TurboStream AutoResearch Loop
Simplified Karpathy-style autonomous experiment runner for llama.cpp configs.
Runs overnight, tests hypotheses, records ledger, commits improvements.

Date: 2026-04-25
Campaign: ANE-TurboStream v3 Physical Ceiling
Model: Qwen3.5-35B-A3B-UD-IQ2_M.gguf
Harness: build-nsg-opt/bin/llama-bench
"""

import json
import os
import subprocess
import time
import sys
import signal
import math

PROJECT = "/Users/manuelmonteiro/Desktop/ANE project"
MODEL = "/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
BENCH = f"{PROJECT}/build-nsg-opt/bin/llama-bench"
RESULTS_DIR = f"{PROJECT}/results/autoresearch"
LEDGER = f"{RESULTS_DIR}/ledger.tsv"
GIT_LOCK = f"{RESULTS_DIR}/.git_running"

# Current best config (from official baseline)
best_tok_s = 26.02
best_config = {"moe_topk": 4, "threads": 4, "ubatch": None, "batch": None}

# Hypothesis queue
hypotheses = [
    {"moe_topk": 4, "threads": 2},
    {"moe_topk": 4, "threads": 6},
    {"moe_topk": 4, "threads": 8},
    {"moe_topk": 4, "ubatch": 64},
    {"moe_topk": 4, "ubatch": 96},
    {"moe_topk": 4, "ubatch": 192},
    {"moe_topk": 4, "ubatch": 256},
    {"moe_topk": 4, "batch": 1024},
    {"moe_topk": 4, "batch": 256},
    {"moe_topk": 3},
    {"moe_topk": 5},
    {"moe_topk": 6},
    {"moe_mode": "stock"},
    {"moe_mode": "resident"},
    {"moe_mode": "slot-bank"},
    {"cpu_strict": 1},
    {"cpu_strict": 0, "poll": 0},
    {"poll": 25},
    {"poll": 75},
    {"fa": 1},
    {"no_kv_offload": 1},
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# Write TSV header if new
if not os.path.exists(LEDGER):
    with open(LEDGER, "w") as f:
        f.write("timestamp\thypothesis_json\ttg128_tok_s\tpp512_tok_s\taction\tnotes\n")

def run_bench(h):
    """Run llama-bench with hypothesis h, return (tg128, pp512) or (None, None) on fail."""
    cmd = [BENCH, "-m", MODEL, "--output", "json"]
    for k, v in h.items():
        flag_map = {
            "moe_topk": "--moe-topk",
            "threads": "-t",
            "ubatch": "--ubatch-size",
            "batch": "--batch-size",
            "moe_mode": "--moe-mode",
            "cpu_strict": "--cpu-strict",
            "poll": "--poll",
            "fa": "--flash-attn",
            "no_kv_offload": "--no-kv-offload",
        }
        flag = flag_map.get(k, f"--{k.replace('_', '-')}")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        out = proc.stdout
        # Parse JSON output from llama-bench
        lines = out.splitlines()
        json_start = None
        for i, l in enumerate(lines):
            if l.strip().startswith("["):
                json_start = i
                break
        if json_start is None:
            return None, None
        data = json.loads("\n".join(lines[json_start:]))
        tg128 = None
        pp512 = None
        for row in data:
            if isinstance(row, dict):
                if row.get("test") == "tg128":
                    tg128 = row.get("t/s", None)
                elif row.get("test") == "pp512":
                    pp512 = row.get("t/s", None)
        return tg128, pp512
    except Exception as e:
        return None, None

def log_result(h, tg128, pp512, action, notes=""):
    with open(LEDGER, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{json.dumps(h)}\t{tg128}\t{pp512}\t{action}\t{notes}\n")

print(f"[AutoResearch] Starting with {len(hypotheses)} hypotheses")
print(f"[AutoResearch] Baseline: tg128={best_tok_s} t/s | config={best_config}")

# Run each hypothesis
for i, h in enumerate(hypotheses):
    print(f"\n[AutoResearch] Hypothesis {i+1}/{len(hypotheses)}: {json.dumps(h)}")
    tg128, pp512 = run_bench(h)
    if tg128 is None:
        log_result(h, None, None, "FAIL", "bench crashed or timed out")
        print(f"  -> FAIL (crash/timeout)")
        continue
    
    gain = ""
    if tg128 > best_tok_s * 1.01:
        action = "KEEP"
        best_tok_s = tg128
        best_config = h.copy()
        notes = f"NEW BEST: {tg128:.2f} t/s"
    elif tg128 >= best_tok_s * 0.99:
        action = "KEEP"
        notes = f"Within noise: {tg128:.2f} t/s"
    else:
        action = "DISCARD"
        notes = f"Regression: {tg128:.2f} t/s"
    
    log_result(h, tg128, pp512, action, notes)
    print(f"  -> {action}: tg128={tg128:.2f} | pp512={pp512:.2f} | {notes}")
    
    # 5-minute wall clock budget per experiment (already enforced by timeout)
    # No sleep needed — llama-bench takes ~30-60s

print(f"\n[AutoResearch] Complete. Best config: {best_config} with {best_tok_s:.2f} t/s")
print(f"[AutoResearch] Ledger: {LEDGER}")
