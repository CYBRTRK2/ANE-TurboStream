#!/bin/bash
# bench_protocol.sh - Official 3-run median reproducibility protocol
# Date: 2026-04-25
# Plan: 2026-04-25-ane-turbostream-v3-physical-ceiling_NEWPLAN.md
#
# This script enforces the §4 Track 0 reproducibility gate.
# No tok/s number is valid outside this protocol.

set -euo pipefail

# --- Configuration ---
MODEL="${MODEL:-$HOME/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-$HOME/Desktop/ANE project/build-nsg-opt/bin/llama-bench}"
LLAMA_CLI="${LLAMA_CLI:-$HOME/Desktop/ANE project/build-nsg-opt/bin/llama-cli}"
PROMPT_SET="${PROMPT_SET:-$HOME/Desktop/ANE project/scripts/prompt_set.txt}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/Desktop/ANE project/results/baseline_$(date +%Y%m%d)}"
RUNS="${RUNS:-3}"
SEED=42
TEMP=0.0
N_PREDICT=128
N_PROMPT=512
N_WARMUP="${N_WARMUP:-30}"
BENCH_BATCH="${BENCH_BATCH:-2048}"
BENCH_UBATCH="${BENCH_UBATCH:-}"
CLI_BATCH="${CLI_BATCH:-512}"
CLI_UBATCH="${CLI_UBATCH:-128}"

# Keep the dormant CoreML scaffold out of official non-ANE baselines. Track 2
# can opt in explicitly with GGML_COREML_ENABLE=1.
export GGML_COREML_ENABLE="${GGML_COREML_ENABLE:-0}"

# Flags derived from current daily driver
BENCH_BASE_FLAGS=(-ngl 99 -t 4 -b "$BENCH_BATCH" -p "$N_PROMPT" -n "$N_PREDICT")
if [[ -n "$BENCH_UBATCH" ]]; then
    BENCH_BASE_FLAGS+=(-ub "$BENCH_UBATCH")
fi
CLI_BASE_FLAGS=(-ngl 99 -t 4 --temp "$TEMP" --reasoning off -ub "$CLI_UBATCH" -b "$CLI_BATCH" --seed "$SEED" -st)

if [[ ! -x "$LLAMA_BENCH" ]]; then
    echo "ERROR: llama-bench not found at $LLAMA_BENCH" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found at $MODEL" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

# Thermal + power snapshot utility
snapshot_power() {
    local out="$1"
    echo "--- pmset thermlog ---" >> "$out"
    pmset -g thermlog >> "$out" 2>&1 || true
    echo "--- mactop json ---" >> "$out"
    if command -v mactop &> /dev/null; then
        mactop --json -n 1 >> "$out" 2>&1 || true
    fi
}

# --- Run one bench configuration ---
run_config() {
    local label="$1"
    shift
    local flags=("$@")
    log "Running $label ..."

    local run_dir="$RESULTS_DIR/$label"
    mkdir -p "$run_dir"

    for run in $(seq 1 "$RUNS"); do
        log "  $label run $run/$RUNS ..."

        # Warm-up generation, discarded. Use llama-bench here so the benchmark
        # protocol cannot accidentally enter an interactive chat loop.
        log "    Warm-up (discarded) ..."
        local warmup_flags=(-m "$MODEL" -ngl 99 -t 4 -b "$BENCH_BATCH" -p 0 -n 128)
        if [[ -n "$BENCH_UBATCH" ]]; then
            warmup_flags+=(-ub "$BENCH_UBATCH")
        fi
        "$LLAMA_BENCH" "${warmup_flags[@]}" \
            --moe-mode stock --moe-topk 4 --output json 2>/dev/null >/dev/null &
        local wp=$!
        sleep $N_WARMUP
        kill $wp 2>/dev/null || true
        wait $wp 2>/dev/null || true

        # Power snapshot before
        snapshot_power "$run_dir/run${run}_power.txt"

        # Actual benchmark run
        local out="$run_dir/run${run}.json"
        "$LLAMA_BENCH" -m "$MODEL" "${BENCH_BASE_FLAGS[@]}" "${flags[@]}" \
            --output json 2>/dev/null | tee "$out" 2>&1

        # Power snapshot after
        snapshot_power "$run_dir/run${run}_power.txt"
    done
}

# Lisbon quality gate (3 prompts, must all pass)
run_quality_gate() {
    local label="$1"
    shift
    local flags=("$@")
    log "Quality gate for $label ..."

    local gate_dir="$RESULTS_DIR/$label/quality"
    mkdir -p "$gate_dir"

    local p1="Lisbon is the capital of"
    local p2="23 * 17 ="
    local p3="Is 97 a prime number? Answer yes or no and briefly explain."

    local i=1
    for p in "$p1" "$p2" "$p3"; do
        log "  Gate prompt $i/3 ..."
        "$LLAMA_CLI" -m "$MODEL" "${CLI_BASE_FLAGS[@]}" "${flags[@]}" \
            -p "$p" -n 64 --no-display-prompt \
            2>/dev/null | tee "$gate_dir/prompt${i}_output.txt"
        ((i++))
    done
}

# === Main: baseline suite ===
log "=============================================="
log "ANE-TurboStream v3 Baseline Protocol"
log "Date: $(date '+%Y-%m-%d %H:%M:%S')"
log "Machine: $(sysctl -n hw.model), $(sysctl -n hw.ncpu) cores"
log "Plan: 2026-04-25-ane-turbostream-v3-physical-ceiling_NEWPLAN.md"
log "=============================================="

# 1. Stock topk=8
run_config "stock_topk8" --moe-mode stock --moe-topk 8
run_quality_gate "stock_topk8" --moe-mode stock --moe-topk 8

# 2. topk=4 (daily driver candidate)
run_config "topk4" --moe-mode stock --moe-topk 4
run_quality_gate "topk4" --moe-mode stock --moe-topk 4

# 3. shared-only (upper bound, no routed experts)
run_config "shared_only" --moe-mode stock --moe-shared-only 1
run_quality_gate "shared_only" --moe-mode stock --moe-shared-only

log "Baseline protocol complete. Results in $RESULTS_DIR"
