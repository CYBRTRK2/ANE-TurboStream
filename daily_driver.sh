#!/bin/bash
# ANE-TurboStream Daily Driver
# Optimized config for Qwen3.5-35B-A3B on M4 Air 16GB
# Verified 2026-04-24: --moe-topk 4 gives +13% over stock, all quality gates pass
# --moe-topk 2 gives +22% but 23x17 wrong. shared-only gives +82% but no routed experts.

set -uo pipefail

MODEL="${MODEL:-$HOME/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf}"
LLAMA_CLI="${LLAMA_CLI:-$HOME/Desktop/ANE project/build-nsg-opt/bin/llama-cli}"

# Core performance flags
NGPU=99
THREADS=4
UBATCH=128
BATCH=512
MOE_TOPK=4

# Quality flags
TEMP=0.0

# Optional: swap model
# MODEL="$HOME/models/Qwen3.5-35B-A3B-Draft-f16.gguf"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

cmd=(
  "$LLAMA_CLI"
  -m "$MODEL"
  -ngl "$NGPU"
  -t "$THREADS"
  -ub "$UBATCH"
  -b "$BATCH"
  --moe-mode stock
  --moe-topk "$MOE_TOPK"
  --reasoning off
  --temp "$TEMP"
  -st
)

log "Running ANE-TurboStream with: ${cmd[*]}"
"${cmd[@]}" "$@"
