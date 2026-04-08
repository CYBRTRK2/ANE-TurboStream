#!/bin/bash
# Experiment 1: Quality Gate for TurboQuant 3-bit KV cache

PROJECT_ROOT="/Users/manuelmonteiro/Desktop/ANE project"
CLI="$PROJECT_ROOT/vendor/anemll-flash-llama.cpp/build/bin/llama-cli"
MODEL="/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
RESULTS="$PROJECT_ROOT/results"

echo "=== Experiment 1: TurboQuant Quality Gate ==="
echo "Testing turbo3_0 KV cache on M4 Air"
echo ""

# Test 1: Lisbon
echo "Test 1: Lisbon (fact check)..."
$CLI -m "$MODEL" -ngl 99 --reasoning off \
    --cache-type-k turbo3 --cache-type-v turbo3 \
    --ctx-size 512 -p "The capital of Portugal is" -n 5 2>&1 | tee $RESULTS/turbo3_test1.txt
if grep -qi "lisbon" $RESULTS/turbo3_test1.txt; then
    echo "✓ PASS: Lisbon found"
    LISBON="OK"
else
    echo "✗ FAIL: Lisbon not found"
    LISBON="FAIL"
fi
echo ""

# Test 2: 345
echo "Test 2: Arithmetic (230+115)..."
$CLI -m "$MODEL" -ngl 99 --reasoning off \
    --cache-type-k turbo3 --cache-type-v turbo3 \
    --ctx-size 512 -p "What is 230 + 115? Answer with just the number:" -n 5 2>&1 | tee $RESULTS/turbo3_test2.txt
if grep -q "345" $RE SULTS/turbo3_test2.txt; then
    echo "✓ PASS: 345 found"
    ARITH="OK"
else
    echo "✗ FAIL: 345 not found"
    ARITH="FAIL"
fi
echo ""

# Test 3: FizzBuzz (short validation)
echo "Test 3: FizzBuzz validation..."
$CLI -m "$MODEL" -ngl 99 --reasoning off \
    --cache-type-k turbo3 --cache-type-v turbo3 \
    --ctx-size 512 -p "Write FizzBuzz in Python for 1 to 10:" -n 30 2>&1 | tee $RESULTS/turbo3_test3.txt
if grep -q "Fizz\|Buzz" $RESULTS/turbo3_test3.txt; then
    echo "✓ PASS: FizzBuzz structure present"
    FIZZ="OK"
else
    echo "✗ FAIL: No FizzBuzz output"
    FIZZ="FAIL"
fi
echo ""

echo "=== Results ==="
echo "Lisbon: $LISBON"
echo "345: $ARITH"
echo "FizzBuzz: $FIZZ"

if [[ "$LISBON" == "OK" && "$ARITH" == "OK" && "$FIZZ" == "OK" ]]; then
    echo ""
    echo "All quality gates PASSED"
    exit 0
else
    echo ""
    echo "Quality gate FAILED"
    exit 1
fi
