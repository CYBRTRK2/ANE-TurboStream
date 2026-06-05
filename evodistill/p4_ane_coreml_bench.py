#!/usr/bin/env python3
"""
Phase 4 Revisit: ANE CoreML Benchmark
=====================================
Proper benchmarking of Apple Neural Engine for batched inference.

Previous Phase 4 conclusion: "ANE is counterproductive for decode"
based on micro-benchmarks of single-vector GEMV (batch=1).

This benchmark tests the CORRECT use case for ANE:
- Batched GEMM at batch sizes 1, 4, 8, 16, 32 (ES fitness evaluation)
- LM-head projection (248320 × 2048) — the decode bottleneck
- Full transformer layer forward pass
- CoreML ANE placement vs GPU placement vs CPU

Attribution:
- Apple Neural Engine: M4 Air 16GB
- CoreML: Apple's official deployment framework
- apple-silicon-internals: Private API reconnaissance (Path B reference)
"""

import coremltools as ct
import numpy as np
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT = Path("/Users/manuelmonteiro/Desktop/ANE project")
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Model dimensions for Qwen3.5-35B-A3B
HIDDEN_DIM = 2048        # d_model
INTER_DIM = 1408         # MLP intermediate (shared expert)
NUM_EXPERTS = 8           # total MoE experts
TOP_K_EXPERTS = 3         # active per token (was 4 in config, 3 in practice for A3B)
VOCAB_SIZE = 248320       # output vocabulary

BATCH_SIZES = [1, 4, 8, 16, 32]
SEQ_LENGTHS = [1, 32, 128]  # 1=decode, 32+=prefill/ES batch


def create_linear_model(input_dim, output_dim, batch_size, name):
    """Create a CoreML model for a linear (GEMM) operation."""
    import torch
    import torch.nn as nn

    class LinearModel(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim, bias=False)

        def forward(self, x):
            return self.linear(x)

    model = LinearModel(input_dim, output_dim)
    model.eval()

    # Trace the model
    example_input = torch.rand(batch_size, input_dim)
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=(batch_size, input_dim))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )

    return mlmodel


def benchmark_coreml_model(mlmodel, input_shape, n_warmup=3, n_runs=20):
    """Benchmark a CoreML model, forcing different compute unit placements."""
    import coremltools as ct

    results = {}

    # Get input/output names from model spec
    input_name = list(mlmodel.input_description.keys())[0]
    input_data = {input_name: np.random.rand(*input_shape).astype(np.float32)}

    # Test different compute unit placements
    compute_units = [
        (ct.ComputeUnit.ALL, "ALL (ANE preferred)"),
        (ct.ComputeUnit.CPU_AND_GPU, "CPU_AND_GPU (no ANE)"),
        (ct.ComputeUnit.CPU_AND_NE, "CPU_AND_NE (no GPU)"),
        (ct.ComputeUnit.CPU_ONLY, "CPU_ONLY"),
    ]

    for compute_unit, label in compute_units:
        try:
            # Re-load model with specific compute unit
            spec = mlmodel.get_spec()
            model_with_cu = ct.models.MLModel(
                spec,
                compute_units=compute_unit,
                minimum_deployment_target=ct.target.macOS15,
            )

            # Warmup
            for _ in range(n_warmup):
                _ = model_with_cu.predict(input_data)

            # Benchmark
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model_with_cu.predict(input_data)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)

            results[label] = {
                "mean_ms": float(np.mean(times)),
                "std_ms": float(np.std(times)),
                "min_ms": float(np.min(times)),
                "median_ms": float(np.median(times)),
                "n_runs": n_runs,
            }
            print(f"  {label}: {np.mean(times):.3f} ± {np.std(times):.3f} ms (median {np.median(times):.3f})")

        except Exception as e:
            results[label] = {"error": str(e)}
            print(f"  {label}: ERROR - {e}")

    return results


def benchmark_lm_head():
    """Benchmark the LM-head projection layer (the decode bottleneck).

    LM-head: [batch, hidden] @ [hidden, vocab] -> [batch, vocab]
    This is the largest single matmul in decode (248320 × 2048).
    """
    print("=" * 70)
    print("BENCHMARK 1: LM-head projection (248320 × 2048)")
    print("=" * 70)

    all_results = {}

    for batch_size in BATCH_SIZES:
        print(f"\n--- Batch size: {batch_size} ---")
        input_shape = (batch_size, HIDDEN_DIM)

        try:
            # Create CoreML model
            mlmodel = create_linear_model(HIDDEN_DIM, VOCAB_SIZE, batch_size,
                                           f"lm_head_b{batch_size}")

            # Check which compute unit ANE gets assigned to
            # (CoreML automatically places ops on ANE if beneficial)
            print(f"  Model created for batch_size={batch_size}")

            results = benchmark_coreml_model(mlmodel, input_shape)
            all_results[f"lm_head_b{batch_size}"] = {
                "input_shape": list(input_shape),
                "weight_shape": [VOCAB_SIZE, HIDDEN_DIM],
                "batch_size": batch_size,
                "placements": results,
            }

        except Exception as e:
            print(f"  ERROR at batch_size={batch_size}: {e}")
            all_results[f"lm_head_b{batch_size}"] = {"error": str(e)}

    return all_results


def benchmark_mlp_layer():
    """Benchmark a single MLP layer (gate_proj + up_proj + down_proj).

    For Qwen3.5-35B-A3B shared expert:
    gate_proj: [2048] -> [1408]
    up_proj: [2048] -> [1408]
    down_proj: [1408] -> [2048]
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Shared expert MLP (gate + up + down projections)")
    print("=" * 70)

    all_results = {}

    import torch
    import torch.nn as nn

    class MLPModel(nn.Module):
        def __init__(self, hidden_dim, inter_dim):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_dim, inter_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, inter_dim, bias=False)
            self.down_proj = nn.Linear(inter_dim, hidden_dim, bias=False)

        def forward(self, x):
            # SwiGLU: gate * up, then down
            return self.down_proj(
                nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
            )

    for batch_size in BATCH_SIZES:
        print(f"\n--- Batch size: {batch_size} ---")
        input_shape = (batch_size, HIDDEN_DIM)

        try:
            model = MLPModel(HIDDEN_DIM, INTER_DIM)
            model.eval()

            example_input = torch.rand(batch_size, HIDDEN_DIM)
            traced_model = torch.jit.trace(model, example_input)

            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=(batch_size, HIDDEN_DIM))],
                outputs=[ct.TensorType(name="output")],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS15,
            )

            print(f"  Model created for batch_size={batch_size}")
            results = benchmark_coreml_model(mlmodel, input_shape)
            all_results[f"mlp_b{batch_size}"] = {
                "input_shape": list(input_shape),
                "hidden_dim": HIDDEN_DIM,
                "inter_dim": INTER_DIM,
                "batch_size": batch_size,
                "placements": results,
            }

        except Exception as e:
            print(f"  ERROR at batch_size={batch_size}: {e}")
            all_results[f"mlp_b{batch_size}"] = {"error": str(e)}

    return all_results


def benchmark_attention_layer():
    """Benchmark a single attention layer (QKV projection + output projection).

    For Qwen3.5-35B-A3B:
    QKV: [2048] -> [3 * 2048] (GQA: num_heads=16, head_dim=128)
    O_proj: [2048] -> [2048]
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Attention QKV + output projection")
    print("=" * 70)

    all_results = {}

    import torch
    import torch.nn as nn

    class AttentionModel(nn.Module):
        def __init__(self, hidden_dim, num_heads=16, head_dim=128):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        def forward(self, x):
            qkv = self.qkv_proj(x)
            # Simplified: just the projections, skip actual attention
            # (attention itself is memory-bound, not compute-bound)
            return self.o_proj(qkv)

    for batch_size in BATCH_SIZES:
        print(f"\n--- Batch size: {batch_size} ---")
        input_shape = (batch_size, HIDDEN_DIM)

        try:
            model = AttentionModel(HIDDEN_DIM)
            model.eval()

            example_input = torch.rand(batch_size, HIDDEN_DIM)
            traced_model = torch.jit.trace(model, example_input)

            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=(batch_size, HIDDEN_DIM))],
                outputs=[ct.TensorType(name="output")],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS15,
            )

            print(f"  Model created for batch_size={batch_size}")
            results = benchmark_coreml_model(mlmodel, input_shape)
            all_results[f"attn_b{batch_size}"] = {
                "input_shape": list(input_shape),
                "hidden_dim": HIDDEN_DIM,
                "num_heads": 16,
                "head_dim": 128,
                "batch_size": batch_size,
                "placements": results,
            }

        except Exception as e:
            print(f"  ERROR at batch_size={batch_size}: {e}")
            all_results[f"attn_b{batch_size}"] = {"error": str(e)}

    return all_results


if __name__ == "__main__":
    print("Phase 4 Revisit: ANE CoreML Benchmark")
    print(f"Hardware: M4 MacBook Air 16GB")
    print(f"Model dimensions: hidden={HIDDEN_DIM}, inter={INTER_DIM}, vocab={VOCAB_SIZE}")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Batch sizes to test: {BATCH_SIZES}")
    print()

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "hardware": "M4 MacBook Air 16GB",
        "model_dims": {
            "hidden_dim": HIDDEN_DIM,
            "inter_dim": INTER_DIM,
            "vocab_size": VOCAB_SIZE,
            "num_experts": NUM_EXPERTS,
            "top_k_experts": TOP_K_EXPERTS,
        },
        "batch_sizes": BATCH_SIZES,
        "benchmarks": {},
    }

    # Benchmark 1: LM-head
    try:
        all_results["benchmarks"]["lm_head"] = benchmark_lm_head()
    except Exception as e:
        print(f"LM-head benchmark failed: {e}")
        all_results["benchmarks"]["lm_head_error"] = str(e)

    # Benchmark 2: MLP layer
    try:
        all_results["benchmarks"]["mlp"] = benchmark_mlp_layer()
    except Exception as e:
        print(f"MLP benchmark failed: {e}")
        all_results["benchmarks"]["mlp_error"] = str(e)

    # Benchmark 3: Attention layer
    try:
        all_results["benchmarks"]["attention"] = benchmark_attention_layer()
    except Exception as e:
        print(f"Attention benchmark failed: {e}")
        all_results["benchmarks"]["attention_error"] = str(e)

    # Save results
    output_path = RESULTS_DIR / "p4_ane_coreml_bench.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 70}")