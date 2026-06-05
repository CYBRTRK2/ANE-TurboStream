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

PROJECT = os.environ.get("ANE_PROJECT", "/Users/manuelmonteiro/Desktop/ANE project")
MODEL = os.environ.get("ANE_MODEL", "/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf")
BENCH = os.environ.get("LLAMA_BENCH", f"{PROJECT}/build-nsg-opt/bin/llama-bench")
RESULTS_DIR = f"{PROJECT}/results/autoresearch"
LEDGER = f"{RESULTS_DIR}/ledger.tsv"
GIT_LOCK = f"{RESULTS_DIR}/.git_running"
BENCH_TIMEOUT = int(os.environ.get("AUTORESEARCH_TIMEOUT", "300"))
REPETITIONS = int(os.environ.get("AUTORESEARCH_REPETITIONS", "3"))
MAX_HYPOTHESES = int(os.environ.get("AUTORESEARCH_MAX", "0"))

# Live control config. AutoResearch compares hypotheses against a control from
# the same thermal/load window instead of a stale historical baseline.
control_config = {"moe_topk": 4, "threads": 4, "ubatch": 512, "batch": 2048, "moe_mode": "stock"}
best_tok_s = float(os.environ.get("AUTORESEARCH_BASELINE_TG", "0"))
best_config = control_config.copy()

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
    """Run llama-bench with hypothesis h, return (tg128, pp512, notes)."""
    cfg = control_config.copy()
    cfg.update(h)

    cmd = [
        BENCH, "-m", MODEL,
        "-p", "512", "-n", "128",
        "-ngl", "99",
        "-t", str(cfg["threads"]),
        "-b", str(cfg["batch"]),
        "-ub", str(cfg["ubatch"]),
        "-r", str(REPETITIONS),
        "--moe-mode", str(cfg["moe_mode"]),
        "--moe-topk", str(cfg["moe_topk"]),
        "--output", "json",
    ]

    for k, v in h.items():
        if k in {"moe_topk", "threads", "ubatch", "batch", "moe_mode"}:
            continue
        flag_map = {
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
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=BENCH_TIMEOUT)
        out = proc.stdout
        stderr_snippet = proc.stderr.strip().replace("\n", " ")[:300]
        # Parse JSON output from llama-bench
        lines = out.splitlines()
        json_start = None
        for i, l in enumerate(lines):
            if l.strip().startswith("["):
                json_start = i
                break
        if json_start is None:
            return None, None, f"no JSON; rc={proc.returncode}; stderr={stderr_snippet}"
        data = json.loads("\n".join(lines[json_start:]))
        tg128 = None
        pp512 = None
        for row in data:
            if isinstance(row, dict):
                # llama-bench JSON identifies tests by n_prompt / n_gen, not "test"
                if row.get("n_gen") == 128:
                    tg128 = row.get("avg_ts", None)
                elif row.get("n_prompt") == 512 and row.get("n_gen") == 0:
                    pp512 = row.get("avg_ts", None)
        if tg128 is None:
            return None, pp512, f"missing tg128; rc={proc.returncode}; stderr={stderr_snippet}"
        return tg128, pp512, f"rc={proc.returncode}; reps={REPETITIONS}"
    except Exception as e:
        print(f"  [run_bench] exception: {e}")
        return None, None, str(e)

def log_result(h, tg128, pp512, action, notes=""):
    with open(LEDGER, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{json.dumps(h)}\t{tg128}\t{pp512}\t{action}\t{notes}\n")

def main():
    global best_tok_s, best_config

    print(f"[AutoResearch] Starting with {len(hypotheses)} hypotheses")
    print(f"[AutoResearch] Control config: {control_config}")
    print(f"[AutoResearch] llama-bench repetitions: {REPETITIONS}")

    if best_tok_s <= 0:
        print("[AutoResearch] Measuring live control baseline ...")
        tg128, pp512, notes = run_bench({})
        if tg128 is None:
            log_result({"control": control_config}, None, pp512, "FAIL", f"control failed: {notes}")
            print(f"[AutoResearch] Control failed: {notes}")
            return
        best_tok_s = tg128
        best_config = control_config.copy()
        log_result({"control": control_config}, tg128, pp512, "CONTROL", notes)
        print(f"[AutoResearch] Live control: tg128={tg128:.2f} | pp512={pp512:.2f} | {notes}")
    else:
        print(f"[AutoResearch] External baseline: tg128={best_tok_s} t/s | config={best_config}")

    queue = hypotheses[:MAX_HYPOTHESES] if MAX_HYPOTHESES > 0 else hypotheses

    # Run each hypothesis
    for i, h in enumerate(queue):
        print(f"\n[AutoResearch] Hypothesis {i+1}/{len(queue)}: {json.dumps(h)}")
        tg128, pp512, bench_notes = run_bench(h)
        if tg128 is None:
            log_result(h, None, pp512, "FAIL", bench_notes)
            print(f"  -> FAIL (crash/timeout)")
            continue

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

        log_result(h, tg128, pp512, action, f"{notes}; {bench_notes}")
        print(f"  -> {action}: tg128={tg128:.2f} | pp512={pp512:.2f} | {notes}")

        # 5-minute wall clock budget per experiment (already enforced by timeout)
        # No sleep needed — llama-bench takes ~30-60s

    print(f"\n[AutoResearch] Complete. Best config: {best_config} with {best_tok_s:.2f} t/s")
    print(f"[AutoResearch] Ledger: {LEDGER}")


if __name__ == "__main__":
    main()
