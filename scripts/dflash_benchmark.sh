#!/bin/bash
# DFlash vs Baseline benchmark for Qwen3.5-35B-A3B on M4 Air 16GB
# Uses dflash-mlx (MLX speculative decoding) vs stock mlx_lm baseline

PROMPT='The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.'

TARGET_MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"
DRAFT_MODEL="z-lab/Qwen3.5-35B-A3B-DFlash"

echo "============================================"
echo "ANE-TurboStream DFlash Benchmark Suite"
echo "Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
echo "Memory: $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $2/1024/1024/1024}')"
echo "Date: $(date)"
echo "============================================"
echo ""

# --- Baseline: stock MLX autoregressive ---
echo ">>> BASELINE: Stock MLX autoregressive"
echo "Model: $TARGET_MODEL"
echo "Prompt tokens: ~128, Max tokens: 512"
echo ""

python3 -c "
import time, json
from mlx_lm import load, stream_generate

model, tokenizer = load('$TARGET_MODEL')

prompt = '''$PROMPT'''
messages = [{'role': 'user', 'content': prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

tok_count = 0
t0 = time.perf_counter()
for token in stream_generate(model, tokenizer, formatted, max_tokens=512):
    tok_count += 1
elapsed = time.perf_counter() - t0

tok_per_sec = tok_count / elapsed
result = {
    'mode': 'baseline',
    'tokens_generated': tok_count,
    'elapsed_sec': round(elapsed, 3),
    'tok_per_sec': round(tok_per_sec, 2)
}
print(json.dumps(result, indent=2))
" 2>&1

echo ""
echo ">>> DFLASH: Speculative decoding"
echo "Target: $TARGET_MODEL"
echo "Draft: $DRAFT_MODEL"
echo ""

dflash --model "$TARGET_MODEL" --draft "$DRAFT_MODEL" --prompt "$PROMPT" --max-tokens 512 2>&1

echo ""
echo "============================================"
echo "Benchmark complete."
echo "============================================"