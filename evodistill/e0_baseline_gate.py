#!/usr/bin/env python3
"""
E0: Baseline Quality Gate for Phase 5 EvoDistill
Runs the frozen model through Lisbon, 345, and FizzBuzz checks.
Logs results to evodistill/runs/baseline_frozen.json

Attribution:
- Inference engine: llama.cpp (anemll-flash-llama.cpp fork)
- SSD self-distillation: Apple Research (arXiv 2604.01193)
- EGGROLL optimizer: @rustane_dev (March 2026)
- LiveCodeBench V6: Apple Research ml-ssd repo
- ANE access: apple-silicon-internals repo
"""

import json
import subprocess
import time
import os
from datetime import datetime

PROJECT = "/Users/manuelmonteiro/Desktop/ANE project"
MODEL = "/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
BIN = f"{PROJECT}/vendor/anemll-flash-llama.cpp/build/bin/llama-completion"
RESULTS_DIR = f"{PROJECT}/evodistill/runs"

os.makedirs(RESULTS_DIR, exist_ok=True)

def run_quality_gate(prompt, n_tokens=256, cache_type="q4_0"):
    """Run the model and capture output."""
    cmd = [
        BIN,
        "-m", MODEL,
        "-ctk", cache_type,
        "-ngl", "99",
        "-n", str(n_tokens),
        "--temp", "0.0",
        "-rea", "off",
        "-p", prompt,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "prompt": prompt,
        "n_tokens": n_tokens,
        "cache_type": cache_type,
    }

# Quality gate prompts
gates = [
    {
        "name": "Lisbon",
        "prompt": "The capital of Portugal is",
        "check": lambda out: "Lisbon" in out or "lisbon" in out.lower(),
        "description": "Factual recall: capital of Portugal"
    },
    {
        "name": "345",
        "prompt": "What is 230 + 115? Answer with the number.",
        "check": lambda out: "345" in out,
        "description": "Arithmetic: 230 + 115 = 345"
    },
    {
        "name": "FizzBuzz",
        "prompt": "Write FizzBuzz in Python for numbers 1 to 20.",
        "check": lambda out: ("fizz" in out.lower() and "buzz" in out.lower()) or ("Fizz" in out and "Buzz" in out) or ("% 3" in out and "% 5" in out),
        "description": "Code generation: FizzBuzz"
    },
]

results = {
    "timestamp": datetime.now().isoformat(),
    "model": os.path.basename(MODEL),
    "phase": "5_E0",
    "gates": {},
    "all_passed": True,
}

print("=" * 60)
print("E0: BASELINE QUALITY GATE — Phase 5 EvoDistill")
print("=" * 60)

for gate in gates:
    print(f"\nRunning gate: {gate['name']} ({gate['description']})...")
    result = run_quality_gate(gate["prompt"])
    # Extract completion from stdout
    # llama-completion outputs: user\n<prompt>\nassistant\n<think>...\n</think>\n<completion>
    output_text = result["stdout"]
    # Strip EOF marker
    output_text = output_text.split("> EOF")[0]
    # Extract just the assistant's response (after "assistant" line)
    lines = output_text.split('\n')
    assistant_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'assistant':
            assistant_start = i + 1
            break
    if assistant_start is not None:
        output_text = '\n'.join(lines[assistant_start:])
    # Strip thinking blocks (Qwen3.5 uses ◁think▷ or <think> tags, or plain "Thinking Process:" lines)
    import re
    # Remove <think>...</think> blocks
    output_text = re.sub(r'<think>.*?</think>\s*', '', output_text, flags=re.DOTALL)
    # Remove ◁think▷...◁/think▷ blocks
    output_text = re.sub(r'◁think▷.*?◁/think▷\s*', '', output_text, flags=re.DOTALL)
    # Remove lines starting with "Thinking Process:" until we hit a blank line or non-thinking content
    output_text = re.sub(r'^Thinking Process:.*?(?=\n\n|\Z)', '', output_text, flags=re.DOTALL)
    output_text = output_text.strip()
    
    # Parse decode speed from stderr (llama-completion prints performance stats)
    tok_per_s = None
    for line in result["stderr"].split("\n"):
        # llama-completion format: "eval time =     606.36 ms /    15 runs   (  40.42 ms per token,   24.74 tokens per second)"
        if "tokens per second" in line or "tok/s" in line:
            try:
                import re
                m = re.search(r'(\d+\.\d+)\s+tokens per second', line)
                if m:
                    tok_per_s = float(m.group(1))
            except:
                pass
    
    passed = gate["check"](output_text)
    if not passed:
        results["all_passed"] = False
    
    gate_result = {
        "prompt": gate["prompt"],
        "passed": passed,
        "description": gate["description"],
        "output_preview": output_text[:200] if output_text else "",
        "tok_per_s": tok_per_s,
    }
    results["gates"][gate["name"]] = gate_result
    
    status = "PASS" if passed else "FAIL"
    print(f"  {gate['name']}: {status}")
    if tok_per_s:
        print(f"  Speed: {tok_per_s:.1f} tok/s")
    print(f"  Output: {output_text[:100]}...")

# Save results
output_path = os.path.join(RESULTS_DIR, "baseline_frozen.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}")
if results["all_passed"]:
    print("ALL QUALITY GATES PASSED — Phase 5 can proceed")
else:
    print("QUALITY GATE FAILED — DO NOT PROCEED")
print(f"Results saved to: {output_path}")
print(f"Decode speed: {tok_per_s:.1f} tok/s")
print(f"Timestamp: {results['timestamp']}")