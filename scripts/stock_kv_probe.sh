#!/bin/bash
# Stock KV Cache Probe Harness
# Task 2 of 2026-04-05-stock-first-kv-path.md
# Measures stock llama.cpp with built-in KV cache types

set -e

PROJECT_ROOT="/Users/manuelmonteiro/Desktop/ANE project"
VENDOR_DIR="$PROJECT_ROOT/vendor/anemll-flash-llama.cpp"
MODEL_PATH="/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
RESULTS_DIR="$PROJECT_ROOT/results"
CLI_BIN="$VENDOR_DIR/build/bin/llama-cli"

# Quality gate prompts
LISBON_PROMPT="The capital of Portugal is"
ARITH_PROMPT="What is 230 + 115?"
FIZZ_PROMPT="Write FizzBuzz in Python for 1 to 20"

# Test configuration
CTX_SIZES=(2048 8192 16384)
CACHE_TYPES=("baseline" "q8_0" "q4_0")

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_quality_gate() {
    local cache_type=$1
    local ctx=$2
    local tag=$3
    
    log "Running quality gate for $cache_type @ ctx=$ctx"
    
    local cache_arg=""
    if [ "$cache_type" != "baseline" ]; then
        cache_arg="--cache-type-k $cache_type --cache-type-v $cache_type"
    fi
    
    # Lisbon check
    local output
    output=$($CLI_BIN -m "$MODEL_PATH" -ngl 99 --reasoning off $cache_arg \
        --ctx-size $ctx -p "$LISBON_PROMPT" -n 5 2>/dev/null | tail -20)
    
    if echo "$output" | grep -qi "lisbon"; then
        log "  ✓ Lisbon check PASSED"
        echo "Lisbon: PASS" >> "$RESULTS_DIR/${tag}_quality.txt"
    else
        log "  ✗ Lisbon check FAILED"
        log "  Output: $output"
        echo "Lisbon: FAIL" >> "$RESULTS_DIR/${tag}_quality.txt"
    fi
    
    # 345 check
    output=$($CLI_BIN -m "$MODEL_PATH" -ngl 99 --reasoning off $cache_arg \
        --ctx-size $ctx -p "$ARITH_PROMPT" -n 5 2>/dev/null | tail -20)
    
    if echo "$output" | grep -q "345"; then
        log "  ✓ 345 check PASSED"
        echo "345: PASS" >> "$RESULTS_DIR/${tag}_quality.txt"
    else
        log "  ✗ 345 check FAILED"
        echo "345: FAIL" >> "$RESULTS_DIR/${tag}_quality.txt"
    fi
}

run_benchmark() {
    local cache_type=$1
    local ctx=$2
    local tag="stock_kv_${cache_type}_ctx${ctx}"
    
    log "Running benchmark: $cache_type @ ctx=$ctx"
    
    local cache_arg=""
    if [ "$cache_type" != "baseline" ]; then
        cache_arg="--cache-type-k $cache_type --cache-type-v $cache_type"
    fi
    
    local raw_file="$RESULTS_DIR/${tag}.raw.txt"
    local summary_file="$RESULTS_DIR/${tag}.summary.md"
    
    # Log exact command
    echo "=== Command ===" > "$raw_file"
    echo "llama-cli -m Qwen3.5-35B-A3B-UD-IQ2_M.gguf -ngl 99 --reasoning off $cache_arg --ctx-size $ctx -p \"[PROMPT]\" -n 128" >> "$raw_file"
    echo "" >> "$raw_file"
    
    # Run llama-bench style benchmark using llama-cli
    # Generate a longer prompt for better measurement
    local prompt="Explain the concept of neural architecture search in machine learning, including:"
    prompt="${prompt} 1) What is the main goal of NAS. 2) Common search strategies used."
    prompt="${prompt} 3) Performance metrics evaluated. 4) Challenges and future directions."
    
    local start_time end_time duration
    start_time=$(date +%s.%N)
    
    { $CLI_BIN -m "$MODEL_PATH" -ngl 99 --reasoning off $cache_arg \
        --ctx-size $ctx -p "$prompt" -n 128 2>&1 || true; } | tee -a "$raw_file"
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc 2>/dev/null || echo "unknown")
    
    # Extract timing info
    echo "" >> "$raw_file"
    echo "=== Timing ===" >> "$raw_file"
    echo "Duration: ${duration}s" >> "$raw_file"
    
    # Create summary
    echo "# Stock KV Probe: $cache_type @ ctx=$ctx" > "$summary_file"
    echo "" >> "$summary_file"
    echo "- Cache type: $cache_type" >> "$summary_file"
    echo "- Context size: $ctx" >> "$summary_file"
    echo "- Duration: ${duration}s" >> "$summary_file"
    echo "- Timestamp: $(date)" >> "$summary_file"
    
    log "  Saved: $raw_file, $summary_file"
}

main() {
    log "Starting Stock KV Probe Harness"
    log "Model: Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
    log "CLI: $CLI_BIN"
    
    mkdir -p "$RESULTS_DIR"
    
    for cache in "${CACHE_TYPES[@]}"; do
        for ctx in "${CTX_SIZES[@]}"; do
            log "---"
            run_benchmark "$cache" "$ctx"
            
            # Quality gate only on first context size per cache type
            if [ "$ctx" = "2048" ]; then
                run_quality_gate "$cache" "$ctx" "stock_kv_${cache}_ctx${ctx}"
            fi
        done
    done
    
    log "---"
    log "All benchmarks complete. Results in $RESULTS_DIR"
}

main "$@"
