#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$ROOT/vendor/anemll-flash-llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
MODEL="${MODEL:-/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf}"
RESULTS_DIR="$ROOT/results"
PROMPT="${PROMPT:-The capital of Portugal is}"

mkdir -p "$RESULTS_DIR"

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DGGML_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_FLASH_MOE_GPU_BANK=ON

cmake --build "$BUILD_DIR" --config Release -j"$(sysctl -n hw.ncpu)" \
  --target llama-cli llama-bench llama-perplexity

{
  echo "DATE=$(date -Iseconds)"
  echo "MODEL=$MODEL"
  echo "COMMAND=$BUILD_DIR/bin/llama-bench -m $MODEL --moe-mode stock -ngl 99 -p 64 -n 32 -r 1 --no-warmup -o md"
} > "$RESULTS_DIR/p0_stock_notes.txt"

"$BUILD_DIR/bin/llama-bench" \
  -m "$MODEL" \
  --moe-mode stock \
  -ngl 99 \
  -p 64 \
  -n 32 \
  -r 1 \
  --no-warmup \
  -o md > "$RESULTS_DIR/p0_stock_baseline.md"

"$BUILD_DIR/bin/llama-cli" \
  -m "$MODEL" \
  --moe-mode stock \
  --seed 123 \
  --temp 0 \
  --reasoning off \
  -ngl 99 \
  -st \
  -n 24 \
  -p "$PROMPT" > "$RESULTS_DIR/p0_stock_smoke.txt"

echo "Phase 0 baseline artifacts written to $RESULTS_DIR"
