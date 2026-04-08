#!/bin/bash
# Phase 1 Production Benchmark: Slot-Bank 32 at real context sizes
# Also re-verify stock baseline at 64K for comparison
set -e

PROJECT_ROOT="/Users/manuelmonteiro/Desktop/ANE project"
VENDOR_DIR="$PROJECT_ROOT/vendor/anemll-flash-llama.cpp"
MODEL_PATH="/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
SIDECAR_PATH="$PROJECT_ROOT/results/sidecar/qwen35"
RESULTS_DIR="$PROJECT_ROOT/results"
CLI_BIN="$VENDOR_DIR/build/bin/llama-cli"
LOG="$RESULTS_DIR/p10_slotbank_bench.log"

mkdir -p "$RESULTS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

# Benchmark function using llama-cli with prompt token counting
# llama-cli outputs eval timing stats including prompt t/s and gen t/s
run_bench() {
    local tag=$1
    shift
    local cmd_args=("$@")
    
    log "=== Running: $tag ==="
    log "Command: $CLI_BIN ${cmd_args[*]}"
    
    local raw_file="$RESULTS_DIR/${tag}.raw.txt"
    local summary_file="$RESULTS_DIR/${tag}.summary.md"
    
    # Run llama-cli, capture output
    # Use --log-disable to avoid stderr noise, -n 128 for generation tokens
    "$CLI_BIN" "${cmd_args[@]}" \
        -n 128 \
        --log-disable 2>&1 | tee "$raw_file"
    
    # Parse timing from output
    local prompt_tps gen_tps
    prompt_tps=$(grep -o 'prompt.*eval.*=.*[0-9.]* tok/s' "$raw_file" 2>/dev/null | grep -o '[0-9.]* tok/s' | head -1 | grep -o '[0-9.]*' || echo "N/A")
    gen_tps=$(grep -o 'generation.*eval.*=.*[0-9.]* tok/s' "$raw_file" 2>/dev/null | grep -o '[0-9.]* tok/s' | head -1 | grep -o '[0-9.]*' || echo "N/A")
    
    # Fallback: try alternative parsing
    if [ "$gen_tps" = "N/A" ]; then
        # llama-cli outputs format: "eval time ... X.XX tok/s"
        gen_tps=$(grep -i 'tok/s' "$raw_file" | grep -i 'eval' | tail -1 | grep -o '[0-9]\+\.[0-9]\+ tok/s' | grep -o '[0-9]\+\.[0-9]\+' || echo "N/A")
    fi
    
    if [ "$prompt_tps" = "N/A" ]; then
        prompt_tps=$(grep -i 'tok/s' "$raw_file" | head -1 | grep -o '[0-9]\+\.[0-9]\+ tok/s' | grep -o '[0-9]\+\.[0-9]\+' || echo "N/A")
    fi
    
    # Quality gate check
    local lisbon_pass=false
    local arith_pass=false
    local output_text
    output_text=$(cat "$raw_file")
    
    if echo "$output_text" | grep -qi "lisbon"; then
        lisbon_pass=true
    fi
    if echo "$output_text" | grep -q "345"; then
        arith_pass=true
    fi
    
    # Write summary
    echo "# Benchmark: $tag" > "$summary_file"
    echo "" >> "$summary_file"
    echo "- date: $(date)" >> "$summary_file"
    echo "- prompt_tps: ${prompt_tps}" >> "$summary_file"
    echo "- gen_tps: ${gen_tps}" >> "$summary_file"
    echo "- lisbon_check: ${lisbon_pass}" >> "$summary_file"
    echo "- arith_check: ${arith_pass}" >> "$summary_file"
    echo "- command: $CLI_BIN ${cmd_args[*]}" >> "$summary_file"
    
    log "  Results: prompt=${prompt_tps} t/s, gen=${gen_tps} t/s"
    log "  Quality: lisbon=${lisbon_pass}, arith=${arith_pass}"
}

log "=== Phase 10: Slot-Bank Production Benchmark ==="
log "Machine: M4 Air 16GB"

# Check prerequisites
if [ ! -f "$CLI_BIN" ]; then
    log "ERROR: llama-cli not found at $CLI_BIN"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    log "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Run 1: Stock baseline @ 64K (re-verify our defended number)
run_bench "p10_stock_q4_ctx65536" \
    -m "$MODEL_PATH" -ngl 99 --reasoning off \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --ctx-size 65536 \
    -p "The capital of Portugal is"

# Run 2: Slot-Bank 32 @ 64K (the key missing benchmark)
run_bench "p10_sb32_ctx65536" \
    -m "$MODEL_PATH" -ngl 99 --reasoning off \
    --moe-mode slot-bank \
    --moe-sidecar "$SIDECAR_PATH" \
    --moe-slot-bank 32 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --ctx-size 65536 \
    -p "The capital of Portugal is"

# Run 3: Slot-Bank 16 @ 64K (lower memory, for comparison)
run_bench "p10_sb16_ctx65536" \
    -m "$MODEL_PATH" -ngl 99 --reasoning off \
    --moe-mode slot-bank \
    --moe-sidecar "$SIDECAR_PATH" \
    --moe-slot-bank 16 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --ctx-size 65536 \
    -p "The capital of Portugal is"

log "=== All benchmarks complete ==="