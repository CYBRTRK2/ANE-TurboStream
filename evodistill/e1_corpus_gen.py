#!/usr/bin/env python3
"""
E1: Corpus Generation — SSD Stage 1 (Apple Research, arXiv 2604.01193)
Generates N samples per prompt from the frozen model at T_train.

Uses the llama-server HTTP endpoint (much faster than spawning llama-completion per sample).

Attribution:
- SSD self-distillation: Apple Research (arXiv 2604.01193)
- EGGROLL optimizer: @rustane_dev (March 2026)
- LiveCodeBench V6: Apple Research ml-ssd repo
- Inference: llama.cpp (anemll-flash-llama.cpp fork)
"""

import json
import os
import re
import time
import yaml
import random
import requests
from datetime import datetime
from pathlib import Path

PROJECT = Path("/Users/manuelmonteiro/Desktop/ANE project")
with open(PROJECT / "evodistill/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Use llama-server HTTP endpoint (model already loaded, no per-sample reload)
SERVER_URL = "http://127.0.0.1:8081/v1/chat/completions"

CORPUS_DIR = PROJECT / CONFIG["corpus"]["output_dir"]
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# --- Prompt Sources ---
# Generate coding prompts similar to LiveCodeBench V6 style
# (In production, these would come from the actual LCB dataset)
PROMPTS = [
    "Write a Python function that finds the longest increasing subsequence in a list.",
    "Implement a function that checks if a binary tree is balanced.",
    "Write a function to find the kth smallest element in an unsorted array.",
    "Implement merge sort in Python with O(n log n) time complexity.",
    "Write a function that computes the edit distance between two strings.",
    "Implement a function to detect a cycle in a linked list.",
    "Write a Python function that finds all anagrams of a word in a dictionary.",
    "Implement a function that solves the N-queens problem.",
    "Write a function to find the maximum subarray sum (Kadane's algorithm).",
    "Implement a binary search function that works on rotated sorted arrays.",
    "Write a Python function to reverse a linked list in place.",
    "Implement a function that validates a binary search tree.",
    "Write a function to find the longest common subsequence of two strings.",
    "Implement a function that finds the shortest path in a weighted graph (Dijkstra).",
    "Write a Python function that counts the number of islands in a 2D grid.",
    "Implement a function that solves the word break problem.",
    "Write a function to find the median of two sorted arrays.",
    "Implement a function that generates all valid parentheses combinations.",
    "Write a Python function that finds the minimum window substring.",
    "Implement a function that computes the power of a number efficiently.",
    "Write a function to serialize and deserialize a binary tree.",
    "Implement a function that finds all permutations of a string.",
    "Write a Python function that solves the trapping rain water problem.",
    "Implement a function that merges overlapping intervals.",
    "Write a function to find the first missing positive integer in an array.",
    "Implement a function that computes the LRU cache eviction policy.",
    "Write a Python function that finds the most frequent element in a stream.",
    "Implement a function that solves the two-sum problem with O(n) time.",
    "Write a function to find the diameter of a binary tree.",
    "Implement a function that builds a Huffman coding tree.",
    "Write a Python function that solves the knapsack problem.",
    "Implement a function that finds the longest palindromic substring.",
    "Write a function to compute the matrix chain multiplication optimization.",
    "Implement a function that performs topological sort on a directed graph.",
    "Write a Python function that finds subsets that sum to a target.",
    "Implement a function that checks if a string is a valid number.",
    "Write a function to find the lowest common ancestor in a binary tree.",
    "Implement a function that counts inversions in an array.",
    "Write a Python function that solves the sliding window maximum problem.",
    "Implement a function that finds the minimum spanning tree (Kruskal).",
    "Write a function to compute the maximum flow in a network (Ford-Fulkerson).",
    "Implement a function that solves the coin change problem.",
    "Write a Python function that finds the longest valid parentheses substring.",
    "Implement a function that performs BFS and DFS on a graph.",
    "Write a function to check if a string can be segmented into dictionary words.",
    "Implement a function that computes the maximum product subarray.",
    "Write a Python function that solves the scheduling problem with intervals.",
    "Implement a function that finds the majority element in an array.",
    "Write a function to compute the combination sum for a given target.",
    "Implement a function that finds the minimum path sum in a grid.",
]

# Expand to desired count by rephrasing
def expand_prompts(base_prompts, target_count):
    """Expand prompt list by adding variations."""
    prefixes = [
        "Write Python code to solve the following problem: ",
        "Implement a solution for: ",
        "Create a Python function that: ",
        "Solve this coding challenge: ",
        "Code a solution in Python for: ",
    ]
    expanded = list(base_prompts)
    idx = 0
    while len(expanded) < target_count:
        base = base_prompts[idx % len(base_prompts)]
        prefix = prefixes[idx % len(prefixes)]
        expanded.append(prefix + base[0].lower() + base[1:])
        idx += 1
    return expanded[:target_count]

N_PROMPTS = CONFIG["corpus"]["n_prompts"]
N_SAMPLES = CONFIG["corpus"]["n_samples"]
T_TRAIN = CONFIG["corpus"]["t_train"]
MAX_TOKENS = CONFIG["corpus"]["max_tokens"]

prompts = expand_prompts(PROMPTS, N_PROMPTS)
random.seed(42)
random.shuffle(prompts)


def clean_completion(raw_output):
    """Clean llama-completion output: strip chat template, thinking blocks, perf lines."""
    output = raw_output.strip()
    # Strip EOF marker
    output = output.split("> EOF")[0].strip()
    # Extract just the assistant's response (after "assistant" line)
    lines = output.split('\n')
    assistant_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'assistant':
            assistant_start = i + 1
            break
    if assistant_start is not None:
        output = '\n'.join(lines[assistant_start:])
    # Strip thinking blocks
    output = re.sub(r'<think>.*?</think>\s*', '', output, flags=re.DOTALL)
    # Strip performance/timing lines
    lines = [l for l in output.split('\n')
             if 'common_perf_print' not in l and
                'tokens per second' not in l and
                'llama_memory' not in l and
                'ggml_metal' not in l and
                l.strip() != '']
    output = '\n'.join(lines).strip()
    return output


def generate_sample(prompt, sample_idx, temp=0.9, max_tokens=512):
    """Generate a single sample from the frozen model using llama-server HTTP."""
    try:
        resp = requests.post(SERVER_URL, json={
            "model": "qwen",
            "messages": [{"role": "user", "content": prompt + " /no_think"}],
            "max_tokens": max_tokens,
            "temperature": temp,
        }, timeout=180)

        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "prompt": prompt}

        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content", "")
        # Strip thinking blocks (Qwen3.5 uses reasoning_content field)
        reasoning = msg.get("reasoning_content", "")
        # If content is empty but reasoning exists, skip (model didn't produce useful output)
        if not content.strip() and reasoning:
            return None

        output = content.strip()
        if not output:
            return None

        # Get token stats
        usage = data.get("usage", {})
        timings = data.get("choices", [{}])[0].get("timings", data.get("timings", {}))

        return {
            "prompt": prompt,
            "completion": output,
            "temperature": temp,
            "sample_idx": sample_idx,
            "max_tokens": max_tokens,
            "timestamp": datetime.now().isoformat(),
            "tokens_generated": usage.get("completion_tokens", 0),
        }
    except requests.exceptions.Timeout:
        return None
    except Exception as e:
        return {"error": str(e), "prompt": prompt}

print("=" * 60)
print("E1: CORPUS GENERATION — SSD Stage 1")
print(f"Attribution: Apple Research (arXiv 2604.01193)")
print(f"Prompts: {N_PROMPTS}, Samples/prompt: {N_SAMPLES}")
print(f"T_train: {T_TRAIN}, Max tokens: {MAX_TOKENS}")
print("=" * 60)

corpus_path = CORPUS_DIR / "corpus.jsonl"
total_samples = N_PROMPTS * N_SAMPLES
completed = 0
failed = 0
start_time = time.time()

# Resume support: count existing samples and determine where to continue
existing_samples = set()
if corpus_path.exists():
    with open(corpus_path, "r") as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                key = f"{sample.get('prompt_idx', '?')}_{sample.get('sample_idx', '?')}"
                existing_samples.add(key)
            except (json.JSONDecodeError, KeyError):
                pass
    completed = len(existing_samples)
    print(f"  Resuming: {completed} existing samples found, {total_samples - completed} remaining")

# Append mode to preserve existing samples
with open(corpus_path, "a" if corpus_path.exists() else "w") as f:
    for p_idx, prompt in enumerate(prompts):
        for s_idx in range(N_SAMPLES):
            # Skip if already generated
            key = f"{p_idx}_{s_idx}"
            if key in existing_samples:
                continue

            sample = generate_sample(prompt, s_idx, temp=T_TRAIN, max_tokens=MAX_TOKENS)
            if sample and "error" not in sample:
                sample["prompt_idx"] = p_idx
                f.write(json.dumps(sample) + "\n")
                completed += 1
            else:
                failed += 1

            completed_so_far = completed + failed
            if completed_so_far % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed_so_far / elapsed * 3600 if elapsed > 0 else 0
                eta_h = (total_samples - completed_so_far) / rate if rate > 0 else 0
                print(f"  [{completed_so_far}/{total_samples}] "
                      f"{completed} ok, {failed} fail, "
                      f"{rate:.0f}/hr, ETA {eta_h:.1f}h")

elapsed = time.time() - start_time
print(f"\nCorpus generation complete:")
print(f"  Completed: {completed}, Failed: {failed}")
print(f"  Elapsed: {elapsed/60:.1f} min")
print(f"  Output: {corpus_path}")

# Save metadata
meta = {
    "timestamp": datetime.now().isoformat(),
    "n_prompts": N_PROMPTS,
    "n_samples": N_SAMPLES,
    "t_train": T_TRAIN,
    "max_tokens": MAX_TOKENS,
    "completed": completed,
    "failed": failed,
    "elapsed_seconds": elapsed,
    "attribution": {
        "ssd": "Apple Research (arXiv 2604.01193)",
        "eggroll": "@rustane_dev",
        "inference": "llama.cpp anemll-flash-llama.cpp fork"
    }
}
with open(CORPUS_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)