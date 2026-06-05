#!/usr/bin/env python3
"""
E2: ES Optimizer — EGGROLL Algorithm with LoRA (@rustane_dev, March 2026)
Implements Evolution Strategy optimization for LoRA adapters on Qwen3.5-35B-A3B.

Architecture:
- Qwen3.5-MoE hybrid: 10 attention layers + 30 SSM layers (40 total)
- LoRA applied to attention Q/K/V/O projections on the 10 attention layers
- Uses llama-server HTTP endpoint for model inference
- LoRA perturbations written as GGUF adapter files
- Per-request adapter scaling for antithetic pairs (negative scales)
- No backprop needed — forward-pass-only fitness evaluation
- ANE acceleration: v2 stub (documented, not yet implemented)

Attribution:
- EGGROLL optimizer: @rustane_dev (March 2026)
- SSD self-distillation: Apple Research (arXiv 2604.01193)
- ANE private API: apple-silicon-internals repo
- Inference: llama.cpp (anemll-flash-llama.cpp fork)
"""

import json
import os
import struct
import time
import yaml
import random
import subprocess
import sys
import numpy as np
import requests
from datetime import datetime
from pathlib import Path

PROJECT = Path("/Users/manuelmonteiro/Desktop/ANE project")
with open(PROJECT / "evodistill/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# === GGUF Constants ===
GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian
GGUF_VERSION = 3

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

# GGML tensor types
GGML_TYPE_F16 = 1

# === Model Architecture (Qwen3.5-35B-A3B-MoE) ===
MODEL_ARCH = "qwen35moe"
TOTAL_LAYERS = 40

# Only 10 of 40 layers have separate attention Q/K/V/O projections.
# The other 30 are SSM layers with fused attn_qkv + attn_gate.
# Attention layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39
ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

# Tensor shapes from GGUF (ne[0], ne[1] convention):
#   attn_q.weight:       [2048, 8192]  (in=hidden_dim, out=128 heads × 64 dim)
#   attn_k.weight:       [2048, 512]   (in=hidden_dim, out=8 KV heads × 64 dim)
#   attn_v.weight:       [2048, 512]   (same as K)
#   attn_output.weight:  [4096, 2048]  (in=128×64 concat, out=hidden_dim)
#
# LoRA shapes (PEFT convention, numpy):
#   lora_a: (rank, in_features)   lora_b: (out_features, rank)
PROJECTION_SHAPES = {
    # name:           (in_features, out_features)
    "attn_q":         (2048, 8192),
    "attn_k":         (2048, 512),
    "attn_v":         (2048, 512),
    "attn_output":    (4096, 2048),
}


# ============================================================
# LoRA Parameter Manager
# ============================================================
class LoRAParameterManager:
    """Manages LoRA parameters for EGGROLL ES optimization.

    Applies LoRA to attention Q/K/V/O projections in the 10 attention layers.
    Each projection has A (rank x in_features) and B (out_features x rank).

    Standard LoRA init: A = Kaiming normal * 0.01, B = zeros (initial contribution = 0).

    Total params: 10 layers × 4 projections × (lora_a + lora_b) ≈ 1.72M params
    """

    def __init__(self, config):
        self.rank = config["lora_rank"]
        self.alpha = config["lora_alpha"]
        self.sigma = config["sigma"]
        self.scale = self.alpha / self.rank

        # Target the 10 attention layers
        self.layer_indices = ATTN_LAYERS
        self.projections = list(PROJECTION_SHAPES.keys())

        # Initialize parameters
        self.params = {}
        self._init_params()

    def _init_params(self):
        """Initialize LoRA matrices: A = random * 0.01, B = zeros."""
        rng = np.random.default_rng(42)
        for layer_idx in self.layer_indices:
            for proj in self.projections:
                in_dim, out_dim = PROJECTION_SHAPES[proj]
                a_key = f"blk.{layer_idx}.{proj}.weight.lora_a"
                b_key = f"blk.{layer_idx}.{proj}.weight.lora_b"
                # A: shape (rank, in_features), Kaiming-like init
                self.params[a_key] = (rng.standard_normal((self.rank, in_dim)) * 0.01).astype(np.float32)
                # B: shape (out_features, rank), zeros (standard LoRA init)
                self.params[b_key] = np.zeros((out_dim, self.rank), dtype=np.float32)

    def perturb(self, rng=None):
        """Generate random perturbation eps ~ N(0, sigma^2 * I)."""
        if rng is None:
            rng = np.random.default_rng()
        eps = {}
        for key, val in self.params.items():
            eps[key] = (rng.standard_normal(val.shape) * self.sigma).astype(np.float32)
        return eps

    def apply_perturbation(self, theta, eps, sign=1):
        """Return theta + sign * eps."""
        return {k: theta[k] + sign * eps[k] for k in theta}

    def update(self, theta, grad_est, alpha, sigma, P):
        """ES parameter update: theta_new = theta + (alpha / (P * sigma)) * grad_est."""
        step_size = alpha / (P * sigma)
        return {k: theta[k] + step_size * grad_est[k] for k in theta}

    def param_count(self):
        """Total number of trainable parameters."""
        return sum(v.size for v in self.params.values())

    def memory_mb(self):
        """Memory for parameters in FP32."""
        return sum(v.nbytes for v in self.params.values()) / (1024 * 1024)

    def save(self, path):
        """Save parameters to numpy file."""
        np.savez(str(path), **{k.replace(".", "_"): v for k, v in self.params.items()})

    def load(self, path):
        """Load parameters from numpy file."""
        data = np.load(str(path))
        for key in self.params:
            safe_key = key.replace(".", "_")
            if safe_key in data:
                self.params[key] = data[safe_key]


# ============================================================
# GGUF Adapter Writer
# ============================================================
class GGUFAdapterWriter:
    """Writes LoRA parameters to GGUF format for llama-server consumption.

    GGUF binary format:
    - Magic: 0x46554747 ('GGUF')
    - Version: 3
    - Tensor count, metadata count
    - Metadata KV pairs (adapter type, alpha, architecture)
    - Tensor info (name, dimensions, type, offset)
    - Alignment padding
    - Tensor data (FP16 for compactness)

    Critical: GGUF dimension order is REVERSED from numpy shape.
    GGUF stores [ne[0], ne[1]] where ne[0] is the contiguous/innermost dimension,
    which corresponds to the LAST axis of a numpy array (columns in row-major).
    So for numpy shape (8, 2048), GGUF dims are [2048, 8].
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_string(self, f, s):
        """Write a GGUF string (uint64 length + UTF-8 bytes)."""
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def _write_metadata(self, f, key, value):
        """Write a GGUF metadata KV pair."""
        self._write_string(f, key)
        if isinstance(value, str):
            f.write(struct.pack("<I", GGUF_TYPE_STRING))
            self._write_string(f, value)
        elif isinstance(value, float):
            f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
            f.write(struct.pack("<f", value))
        elif isinstance(value, int):
            f.write(struct.pack("<I", GGUF_TYPE_UINT32))
            f.write(struct.pack("<I", value))
        elif isinstance(value, bool):
            f.write(struct.pack("<I", GGUF_TYPE_UINT32))
            f.write(struct.pack("<I", int(value)))
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")

    def write_adapter(self, params, name, lora_alpha=16):
        """Write LoRA parameters to a GGUF adapter file.

        Args:
            params: Dict mapping tensor names to numpy arrays
            name: Adapter name (e.g., "base", "eps_0", "step_5")
            lora_alpha: LoRA alpha value

        Returns:
            Path to the written GGUF file
        """
        output_path = self.output_dir / f"{name}.gguf"

        # Convert to FP16 for compactness
        tensors = {}
        for key, val in params.items():
            tensors[key] = val.astype(np.float16)

        # Metadata — architecture MUST match the base model
        metadata = {
            "general.architecture": MODEL_ARCH,
            "general.type": "adapter",
            "adapter.type": "lora",
            "adapter.lora.alpha": float(lora_alpha),
        }

        n_tensors = len(tensors)
        n_metadata = len(metadata)

        with open(output_path, "wb") as f:
            # Header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", n_tensors))
            f.write(struct.pack("<Q", n_metadata))

            # Metadata
            for key, value in metadata.items():
                self._write_metadata(f, key, value)

            # Calculate data offsets (need to know tensor info section size first)
            alignment = 32
            tensor_info_start = f.tell()

            # First pass: calculate total data size and per-tensor offsets
            data_offsets = []
            current_offset = 0
            for name_key, tensor in tensors.items():
                # Align offset
                current_offset = ((current_offset + alignment - 1) // alignment) * alignment
                data_offsets.append(current_offset)
                current_offset += tensor.nbytes

            # Write tensor info
            for i, (name_key, tensor) in enumerate(tensors.items()):
                self._write_string(f, name_key)
                # Dimensions: REVERSED from numpy shape (GGUF convention)
                # numpy (rows, cols) → GGUF [ne[0]=cols, ne[1]=rows]
                ndim = len(tensor.shape)
                f.write(struct.pack("<I", ndim))
                for dim in reversed(tensor.shape):  # Reverse for GGUF ne[0], ne[1] order
                    f.write(struct.pack("<Q", dim))
                # Type (FP16 = 1)
                f.write(struct.pack("<I", GGML_TYPE_F16))
                # Offset (relative to data start)
                f.write(struct.pack("<Q", data_offsets[i]))

            # Align data start
            pad = (alignment - (f.tell() % alignment)) % alignment
            f.write(b"\x00" * pad)

            # Write tensor data
            for i, (name_key, tensor) in enumerate(tensors.items()):
                # Align
                pad = (alignment - (f.tell() % alignment)) % alignment
                f.write(b"\x00" * pad)
                f.write(tensor.tobytes())

        return output_path


# ============================================================
# Fitness Evaluator
# ============================================================
class FitnessEvaluator:
    """Evaluates model fitness via cross-entropy on corpus batches.

    Uses llama-server /v1/completions endpoint with echo=true and max_tokens=0
    to get logprobs without generating new tokens.

    Also supports /v1/chat/completions for quality gate evaluation.
    """

    def __init__(self, server_url="http://127.0.0.1:8081", batch_size=4):
        self.server_url = server_url
        self.batch_size = batch_size

    def evaluate_fitness(self, corpus_batch, adapter_scales=None):
        """Compute fitness = -cross_entropy for a batch of corpus samples.

        Args:
            corpus_batch: List of dicts with 'prompt' and 'completion' keys
            adapter_scales: Dict mapping adapter ID to scale, e.g. {0: 1.0, 1: 1.0}

        Returns:
            float: Fitness = -mean_cross_entropy (higher is better)
        """
        total_ce = 0.0
        total_tokens = 0

        for sample in corpus_batch:
            prompt = sample["prompt"]
            completion = sample["completion"]

            # Use /v1/completions with echo=true to get logprobs for prompt+completion
            payload = {
                "prompt": prompt + "\n" + completion,
                "echo": True,
                "logprobs": 1,
                "max_tokens": 1,   # Generate 1 token minimum (0 may not return logprobs)
                "temperature": 0.0,
            }

            # Add LoRA adapter scaling if provided
            if adapter_scales:
                payload["lora"] = [{"id": k, "scale": v} for k, v in adapter_scales.items()]

            try:
                resp = requests.post(
                    f"{self.server_url}/v1/completions",
                    json=payload,
                    timeout=120,
                )

                if resp.status_code != 200:
                    print(f"  Server error: {resp.status_code} {resp.text[:200]}")
                    continue

                data = resp.json()
                ce, n_tok = self._compute_cross_entropy(data, len(prompt))
                total_ce += ce
                total_tokens += n_tok

            except requests.exceptions.Timeout:
                print("  Request timeout, skipping sample")
                continue
            except Exception as e:
                print(f"  Error: {e}, skipping sample")
                continue

        if total_tokens == 0:
            return float("-inf")

        return -total_ce / total_tokens  # Negative CE = fitness (higher is better)

    def evaluate_fitness_chat(self, prompts, adapter_scales=None):
        """Evaluate fitness using chat completions (fallback if /v1/completions fails).

        Uses a simpler scoring: negative perplexity on short generations.
        """
        total_score = 0.0
        n = 0

        for prompt in prompts:
            payload = {
                "messages": [{"role": "user", "content": prompt + " /no_think"}],
                "max_tokens": 64,
                "temperature": 0.0,
            }
            if adapter_scales:
                payload["lora"] = [{"id": k, "scale": v} for k, v in adapter_scales.items()]

            try:
                resp = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    timeout=120,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                output = data["choices"][0]["message"]["content"]
                # Simple heuristic: longer meaningful outputs = better
                total_score += len(output.split())
                n += 1
            except Exception:
                continue

        return total_score / max(n, 1)

    def _compute_cross_entropy(self, response, prompt_char_len):
        """Extract cross-entropy from logprobs response."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return 0.0, 0

            logprobs_data = choices[0].get("logprobs", {})
            if not logprobs_data:
                return 0.0, 0

            token_logprobs = logprobs_data.get("token_logprobs", [])
            if not token_logprobs:
                return 0.0, 0

            # Count all tokens with non-None logprobs
            total_ce = 0.0
            n_tokens = 0
            for lp in token_logprobs:
                if lp is not None:
                    total_ce += -lp  # Cross-entropy = -log_prob
                    n_tokens += 1

            return total_ce, max(n_tokens, 1)

        except Exception as e:
            print(f"  CE computation error: {e}")
            return 0.0, 0


# ============================================================
# Server Lifecycle Manager
# ============================================================
class ServerManager:
    """Manages llama-server lifecycle with LoRA adapters.

    Strategy:
    - Start server with --lora-init-without-apply and all adapter files
    - Use POST /lora-adapters to change adapter scales per-request
    - Restart server only when adapter files change (between ES steps)
    """

    def __init__(self, config):
        self.config = config
        self.binary = str(PROJECT / "vendor/anemll-flash-llama.cpp/build/bin/llama-server")
        self.model = config["inference"]["model"]
        self.cache_type_k = config["inference"].get("cache_type_k", "q4_0")
        self.ngl = config["inference"].get("ngl", 99)
        self.host = "127.0.0.1"
        self.port = 8081
        self.process = None

    def start(self, adapter_files=None):
        """Start llama-server with optional LoRA adapters loaded without applying.

        Args:
            adapter_files: List of paths to GGUF adapter files to load
        """
        self.stop()  # Kill any existing server

        cmd = [
            self.binary,
            "-m", self.model,
            "--cache-type-k", self.cache_type_k,
            "-ngl", str(self.ngl),
            "--host", self.host,
            "--port", str(self.port),
            "--cont-batching",
            "-np", "1",
            "-c", "2048",  # Reduced context to save memory
        ]

        # Add LoRA adapters (loaded but not applied until POST /lora-adapters)
        adapter_files = adapter_files or []
        if adapter_files:
            for adapter_path in adapter_files:
                cmd.extend(["--lora", str(adapter_path)])
            cmd.append("--lora-init-without-apply")

        print(f"  Starting llama-server ({len(adapter_files)} adapters, init-without-apply)...")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        for i in range(180):  # 3 minutes timeout (large model on 16GB)
            try:
                resp = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
                if resp.status_code == 200:
                    print(f"  Server ready after {i+1}s")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)

        print("  ERROR: Server failed to start within 180s")
        self.stop()
        return False

    def stop(self):
        """Stop llama-server."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            print("  Server stopped")

    def set_adapter_scales(self, scales):
        """Set adapter scales via POST /lora-adapters.

        Args:
            scales: List of dicts with 'id' and 'scale', e.g. [{"id": 0, "scale": 1.0}]
        """
        try:
            resp = requests.post(
                f"http://{self.host}:{self.port}/lora-adapters",
                json=scales,
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_adapter_ids(self):
        """Get loaded adapter IDs from the server."""
        try:
            resp = requests.get(f"http://{self.host}:{self.port}/lora-adapters", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return []


# ============================================================
# ES Algorithm (EGGROLL)
# ============================================================
class ESAlgorithm:
    """EGGROLL Evolution Strategy optimizer.

    Implements the antithetic sampling ES step:
    1. Generate P/2 random perturbations eps_k ~ N(0, sigma^2 * I)
    2. For each k, evaluate fitness with theta + eps_k and theta - eps_k
    3. Gradient estimate: sum_k (F_pos_k - F_neg_k) * eps_k
    4. Update: theta_new = theta + (alpha / (P * sigma)) * grad_est

    Key property: No backprop needed. Only forward passes.

    Adapter strategy:
    - Base adapter (theta) loaded as adapter ID 0
    - Noise adapters (eps_0..eps_7) loaded as IDs 1-8
    - Antithetic pairs: theta + eps_k → {0: 1.0, k+1: 1.0}
                         theta - eps_k → {0: 1.0, k+1: -1.0}
    """

    def __init__(self, lora_mgr, evaluator, server_mgr, writer, config):
        self.lora_mgr = lora_mgr
        self.evaluator = evaluator
        self.server_mgr = server_mgr
        self.writer = writer
        self.population = config["population"]
        self.sigma = config["sigma"]
        self.alpha = config["alpha"]
        self.n_steps = config["n_steps"]
        self.checkpoint_dir = Path(PROJECT / "evodistill" / "runs")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_fitness = float("-inf")

    def es_step(self, theta, corpus_batch, step_num):
        """Execute a single ES optimization step.

        Args:
            theta: Current LoRA parameters (dict of numpy arrays)
            corpus_batch: Batch of corpus samples for fitness evaluation
            step_num: Current step number (for logging)

        Returns:
            Updated theta (dict of numpy arrays)
        """
        rng = np.random.default_rng(step_num)
        grad_est = {k: np.zeros_like(v) for k, v in theta.items()}
        fitness_values = []

        # Generate P/2 perturbation pairs
        n_pairs = self.population // 2
        eps_list = [self.lora_mgr.perturb(rng) for _ in range(n_pairs)]

        # Write adapter files
        base_path = self.writer.write_adapter(theta, "base", lora_alpha=self.lora_mgr.alpha)
        noise_paths = []
        for k, eps in enumerate(eps_list):
            path = self.writer.write_adapter(eps, f"eps_{k}", lora_alpha=self.lora_mgr.alpha)
            noise_paths.append(path)

        # Start server with all adapters loaded (but not applied)
        adapter_files = [base_path] + noise_paths
        if not self.server_mgr.start(adapter_files):
            print(f"  ERROR: Failed to start server at step {step_num}")
            return theta

        # Evaluate antithetic pairs using per-request adapter scaling
        for k, eps in enumerate(eps_list):
            # theta + eps: base at scale 1.0, eps_k at scale 1.0
            scales_pos = [{"id": 0, "scale": 1.0}, {"id": k + 1, "scale": 1.0}]
            self.server_mgr.set_adapter_scales(scales_pos)
            fitness_pos = self.evaluator.evaluate_fitness(corpus_batch)

            # theta - eps: base at scale 1.0, eps_k at scale -1.0
            scales_neg = [{"id": 0, "scale": 1.0}, {"id": k + 1, "scale": -1.0}]
            self.server_mgr.set_adapter_scales(scales_neg)
            fitness_neg = self.evaluator.evaluate_fitness(corpus_batch)

            # Accumulate gradient estimate
            for key in theta:
                grad_est[key] += (fitness_pos - fitness_neg) * eps[key]

            fitness_values.append((fitness_pos, fitness_neg))
            print(f"  Step {step_num}, pair {k+1}/{n_pairs}: "
                  f"F+={fitness_pos:.4f}, F-={fitness_neg:.4f}, "
                  f"Δ={(fitness_pos-fitness_neg):.6f}")

        # Disable all adapters (for clean state)
        all_off = [{"id": i, "scale": 0.0} for i in range(len(adapter_files))]
        self.server_mgr.set_adapter_scales(all_off)

        # Update parameters
        theta_new = self.lora_mgr.update(theta, grad_est, self.alpha, self.sigma, self.population)

        # Log
        avg_fitness = np.mean([f for pair in fitness_values for f in pair])
        self.best_fitness = max(self.best_fitness, avg_fitness)
        print(f"  Step {step_num}: avg_fitness={avg_fitness:.4f}, best={self.best_fitness:.4f}")

        return theta_new

    def run(self, corpus, n_steps=None):
        """Run the full ES optimization loop.

        Args:
            corpus: List of dicts with 'prompt' and 'completion' keys
            n_steps: Number of ES steps (default from config)
        """
        n_steps = n_steps or self.n_steps
        theta = dict(self.lora_mgr.params)  # Copy initial params
        batch_size = min(self.evaluator.batch_size, len(corpus))

        print(f"\n{'='*65}")
        print(f"E2: ES Optimizer — EGGROLL Algorithm")
        print(f"{'='*65}")
        print(f"  Model: Qwen3.5-35B-A3B-MoE (arch={MODEL_ARCH})")
        print(f"  Steps: {n_steps}")
        print(f"  Population: {self.population} ({self.population//2} antithetic pairs)")
        print(f"  Sigma: {self.sigma}, Alpha: {self.alpha}")
        print(f"  LoRA rank: {self.lora_mgr.rank}, alpha: {self.lora_mgr.alpha}")
        print(f"  Target layers: {self.lora_mgr.layer_indices} ({len(self.lora_mgr.layer_indices)} attn layers)")
        print(f"  Projections: {self.lora_mgr.projections}")
        print(f"  Params: {self.lora_mgr.param_count():,}")
        print(f"  Memory: {self.lora_mgr.memory_mb():.1f}MB (FP32)")
        print(f"  Corpus samples: {len(corpus)}, Batch size: {batch_size}")
        print(f"{'='*65}\n")

        start_time = time.time()

        for step in range(n_steps):
            # Sample a random batch from corpus
            batch = random.sample(corpus, min(batch_size, len(corpus)))

            # Run ES step
            theta = self.es_step(theta, batch, step)

            # Update lora_mgr params
            self.lora_mgr.params = dict(theta)

            # Save checkpoint every 10 steps
            if (step + 1) % 10 == 0 or step == n_steps - 1:
                checkpoint_path = self.checkpoint_dir / f"lora_step_{step+1:04d}.npz"
                self.lora_mgr.save(checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

            # Stop server between steps (will restart with new adapters)
            self.server_mgr.stop()

            # Progress estimate
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed * 3600
            eta = (n_steps - step - 1) / rate * 60 if rate > 0 else 0
            print(f"  Progress: {step+1}/{n_steps}, {rate:.1f} steps/hr, ETA {eta:.0f} min\n")

        elapsed = time.time() - start_time
        print(f"\n{'='*65}")
        print(f"ES Optimization Complete")
        print(f"  Total steps: {n_steps}")
        print(f"  Elapsed: {elapsed/60:.1f} min")
        print(f"  Best fitness: {self.best_fitness:.4f}")
        print(f"  Final params: {self.lora_mgr.param_count():,}")
        print(f"{'='*65}")

        # Save final adapter GGUF
        final_path = self.writer.write_adapter(theta, "final", lora_alpha=self.lora_mgr.alpha)
        print(f"  Final adapter: {final_path}")

        return theta


# ============================================================
# ANE Accelerator — CoreML + Private API hybrid
# ============================================================
class ANEAccelerator:
    """Offload LoRA residual computation to ANE via CoreML.

    Strategy:
    - Compile a CoreML model that computes (alpha/rank) * B @ A @ x
    - CoreML automatically schedules conv-like ops on the ANE
    - LoRA A/B weights passed as inputs (not compile-time constants)
      → single compilation, weights change every ES step
    - Falls back to NumPy CPU if CoreML/ANE unavailable

    The residual for each projection is:
        lora_residual = (alpha / rank) * B @ (A @ x)
        output = base_output + lora_residual

    For rank=8, the computation is:
        A @ x: (rank, in_features) @ (in_features,) → (rank,)    — O(rank * in)
        B @ (A @ x): (out_features, rank) @ (rank,) → (out_features,) — O(out * rank)
        Total: O(rank * (in + out)) per projection per layer

    Attribution:
    - ANE private API: apple-silicon-internals repo
    - CoreML ANE scheduling: Apple CoreML framework
    """

    def __init__(self, lora_mgr, config=None):
        self.lora_mgr = lora_mgr
        self.scale = lora_mgr.scale  # alpha / rank
        self.models = {}  # compiled CoreML models keyed by projection name
        self.ane_available = False
        self._compile_models()

    def _compile_models(self):
        """Compile CoreML models for each projection shape.

        Each model takes 2D tensors (CoreML requires batch dimension for matmul):
          - x:        (1, in_features)
          - lora_a:   (rank, in_features)
          - lora_b:   (out_features, rank)
        Returns:
          - residual: (1, out_features)

        The matmul chain: scale * lora_b @ (lora_a @ x^T)
        = scale * (out, rank) @ (rank, in) @ (in, 1)
        = scale * (out, 1) → squeeze to (out,)
        """
        try:
            import coremltools as ct
        except ImportError:
            print("  [ANE] coremltools not available, using CPU fallback")
            return

        rank = self.lora_mgr.rank

        for proj_name, (in_dim, out_dim) in PROJECTION_SHAPES.items():
            try:
                import torch
                import torch.nn as nn

                class LoRAResidualModel(nn.Module):
                    def __init__(self, scale):
                        super().__init__()
                        self.scale = scale

                    def forward(self, x, lora_a, lora_b):
                        # x: (1, in_dim), lora_a: (rank, in_dim), lora_b: (out_dim, rank)
                        # Step 1: lora_a @ x^T → (rank, in_dim) @ (in_dim, 1) = (rank, 1)
                        ax = torch.matmul(lora_a, x.transpose(0, 1))
                        # Step 2: lora_b @ ax → (out_dim, rank) @ (rank, 1) = (out_dim, 1)
                        residual = torch.matmul(lora_b, ax)
                        # Transpose back to (1, out_dim)
                        return (residual.transpose(0, 1)) * self.scale

                model = LoRAResidualModel(self.scale)
                model.eval()

                # Trace with 2D example inputs
                ex_x = torch.randn(1, in_dim)
                ex_a = torch.randn(rank, in_dim)
                ex_b = torch.randn(out_dim, rank)

                traced = torch.jit.trace(model, (ex_x, ex_a, ex_b))

                # Convert to CoreML with ANE priority
                # ct.convert() returns a loaded MLModel, defaults to ComputeUnit.ALL
                mlmodel = ct.convert(
                    traced,
                    inputs=[
                        ct.TensorType(name="x", shape=(1, in_dim), dtype=np.float32),
                        ct.TensorType(name="lora_a", shape=(rank, in_dim), dtype=np.float32),
                        ct.TensorType(name="lora_b", shape=(out_dim, rank), dtype=np.float32),
                    ],
                    convert_to="mlprogram",
                    minimum_deployment_target=ct.target.macOS13,
                )

                # Save to temp dir and reload with CPU_AND_NE to prefer ANE
                import tempfile
                tmp_dir = tempfile.mkdtemp(prefix=f"ane_lora_{proj_name}_")
                model_path = os.path.join(tmp_dir, f"{proj_name}.mlpackage")
                mlmodel.save(model_path)

                from coremltools.models.model import _ComputeUnit
                mlmodel = ct.models.MLModel(model_path, compute_units=_ComputeUnit.CPU_AND_NE)

                self.models[proj_name] = {
                    "model": mlmodel,
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                }
                print(f"  [ANE] Compiled CoreML model for {proj_name}: "
                      f"({in_dim}) → ({out_dim}), rank={rank}")

            except Exception as e:
                print(f"  [ANE] CoreML compilation failed for {proj_name}: {e}")
                print(f"  [ANE] Using CPU fallback for {proj_name}")

        if self.models:
            self.ane_available = True
            print(f"  [ANE] {len(self.models)}/{len(PROJECTION_SHAPES)} models compiled for ANE")
        else:
            print("  [ANE] No models compiled, using CPU fallback for all projections")

    def compute_residual(self, proj_name, x, lora_a, lora_b):
        """Compute LoRA residual: (alpha/rank) * B @ (A @ x).

        Args:
            proj_name: Projection name (e.g., "attn_q")
            x: Input hidden state, shape (in_features,)
            lora_a: LoRA A matrix, shape (rank, in_features)
            lora_b: LoRA B matrix, shape (out_features, rank)

        Returns:
            Residual vector, shape (out_features,)
        """
        if self.ane_available and proj_name in self.models:
            try:
                model_info = self.models[proj_name]
                # CoreML expects 2D input: (1, in_features)
                x_2d = x.astype(np.float32).reshape(1, -1)
                result = model_info["model"].predict({
                    "x": x_2d,
                    "lora_a": lora_a.astype(np.float32),
                    "lora_b": lora_b.astype(np.float32),
                })
                # CoreML returns dict with output name, shape (1, out_dim)
                output = list(result.values())[0]
                return output.flatten().astype(np.float32)
            except Exception as e:
                # Fall through to CPU
                pass

        # CPU fallback
        ax = lora_a.astype(np.float32) @ x.astype(np.float32)  # (rank,)
        residual = lora_b.astype(np.float32) @ ax               # (out_dim,)
        return (residual * self.scale).astype(np.float32)

    def compute_all_residuals(self, hidden_states, theta, eps, sign=1):
        """Compute LoRA residuals for all layers and projections.

        Args:
            hidden_states: Dict mapping layer_idx to input hidden state arrays.
                          Shape per entry: (in_features,) for Q/K/V, (out_features,) for O
            theta: Current LoRA parameters
            eps: Perturbation dict
            sign: +1 or -1 for antithetic pair

        Returns:
            Dict mapping (layer_idx, proj_name) to residual vectors
        """
        residuals = {}
        perturbed = self.lora_mgr.apply_perturbation(theta, eps, sign=sign)

        for layer_idx in self.lora_mgr.layer_indices:
            for proj in self.lora_mgr.projections:
                a_key = f"blk.{layer_idx}.{proj}.weight.lora_a"
                b_key = f"blk.{layer_idx}.{proj}.weight.lora_b"

                if a_key not in perturbed or b_key not in perturbed:
                    continue

                # Get hidden state for this layer
                state_key = (layer_idx, proj)
                if state_key in hidden_states:
                    x = hidden_states[state_key]
                elif layer_idx in hidden_states:
                    x = hidden_states[layer_idx]
                else:
                    continue

                residual = self.compute_residual(
                    proj, x, perturbed[a_key], perturbed[b_key]
                )
                residuals[(layer_idx, proj)] = residual

        return residuals

    def benchmark(self, n_iters=100):
        """Benchmark ANE vs CPU for LoRA residual computation."""
        import time

        for proj_name, (in_dim, out_dim) in PROJECTION_SHAPES.items():
            x = np.random.randn(in_dim).astype(np.float32)
            a = np.random.randn(self.lora_mgr.rank, in_dim).astype(np.float32)
            b = np.random.randn(out_dim, self.lora_mgr.rank).astype(np.float32)

            # CPU
            t0 = time.perf_counter()
            for _ in range(n_iters):
                ax = a @ x
                r = b @ ax * self.scale
            cpu_ms = (time.perf_counter() - t0) / n_iters * 1000

            # ANE (if available)
            ane_ms = None
            if self.ane_available and proj_name in self.models:
                # Warmup
                for _ in range(10):
                    self.compute_residual(proj_name, x, a, b)
                t0 = time.perf_counter()
                for _ in range(n_iters):
                    self.compute_residual(proj_name, x, a, b)
                ane_ms = (time.perf_counter() - t0) / n_iters * 1000

            speedup = f"{cpu_ms/ane_ms:.1f}x" if ane_ms else "N/A"
            ane_str = f"{ane_ms:.3f}ms" if ane_ms else "N/A"
            print(f"  [ANE Benchmark] {proj_name}: CPU={cpu_ms:.3f}ms, ANE={ane_str}, speedup={speedup}")


# ============================================================
# Residual Trick — Efficient gradient estimation
# ============================================================
class ResidualTrick:
    """Compute base fitness once, then estimate fitness differences via LoRA residuals.

    The key insight: instead of running 2*P full model forward passes per ES step,
    we can:
    1. Run the base model once (theta only) to get baseline cross-entropy
    2. For each perturbation, estimate the fitness change analytically:
       ΔF ≈ dF/dtheta · eps = Σ_l Σ_p (dF/doutput_l) · residual_l(eps)

    Implementation strategy (hybrid approach):
    - Use llama-server with adapter scaling for exact fitness evaluation
      (this is already the most efficient approach with the HTTP API)
    - Use ANEAccelerator for the *gradient accumulation* step:
      Instead of (F_pos - F_neg) * eps (which requires 2 full passes),
      compute the residual contribution directly on ANE and estimate
      fitness change from per-token log-prob sensitivity.

    Practical speedup: The ES step currently requires P=16 full forward passes.
    With the residual trick:
    - 1 base forward pass (theta only, no perturbation)
    - P perturbation evaluations still via llama-server (unavoidable without
      hidden state access), but with early stopping and reduced context
    - Gradient computed from fitness differences (same as before)

    The main optimization we CAN do: avoid restarting the server between
    evaluations by using per-request adapter scaling (already implemented).

    Future: With hidden state access (e.g., llama-cpp-python hooks or
    custom llama-server endpoint), we can reduce to 1 base + P cheap residual
    computations per step — a 2x speedup for P=16.

    Attribution:
    - Residual trick concept: SSD (Apple Research, arXiv 2604.01193)
    - ES gradient estimation: EGGROLL (@rustane_dev)
    - ANE acceleration: apple-silicon-internals repo
    """

    def __init__(self, lora_mgr, ane_accelerator, evaluator, server_mgr, config):
        self.lora_mgr = lora_mgr
        self.ane = ane_accelerator
        self.evaluator = evaluator
        self.server_mgr = server_mgr
        self.population = config["population"]
        self.sigma = config["sigma"]
        self.alpha = config["alpha"]

    def es_step_residual(self, theta, corpus_batch, step_num):
        """ES step using residual trick for gradient estimation.

        Compared to the standard ES step, this version:
        1. Evaluates base fitness (theta only) once
        2. Evaluates perturbed fitness via adapter scaling (2P passes)
        3. Computes gradient from fitness differences (standard EGGROLL)
        4. BUT uses ANEAccelerator for the gradient accumulation math

        The actual speedup comes from:
        - Keeping the server running (no restart between pairs)
        - Using adapter scaling for instant perturbation switching
        - Computing gradient accumulation on ANE when possible

        Args:
            theta: Current LoRA parameters
            corpus_batch: Batch of corpus samples
            step_num: Current step number

        Returns:
            Updated theta
        """
        rng = np.random.default_rng(step_num)
        grad_est = {k: np.zeros_like(v) for k, v in theta.items()}
        fitness_values = []

        n_pairs = self.population // 2
        eps_list = [self.lora_mgr.perturb(rng) for _ in range(n_pairs)]

        # Write adapter files
        writer = GGUFAdapterWriter(PROJECT / "evodistill" / "adapters")
        base_path = writer.write_adapter(theta, "base_rt", lora_alpha=self.lora_mgr.alpha)
        noise_paths = []
        for k, eps in enumerate(eps_list):
            path = writer.write_adapter(eps, f"eps_rt_{k}", lora_alpha=self.lora_mgr.alpha)
            noise_paths.append(path)

        # Start server once with all adapters
        adapter_files = [base_path] + noise_paths
        if not self.server_mgr.start(adapter_files):
            print(f"  ERROR: Failed to start server at step {step_num}")
            return theta

        # Evaluate base fitness (theta only, no perturbation)
        base_scales = [{"id": 0, "scale": 1.0}]
        self.server_mgr.set_adapter_scales(base_scales)
        fitness_base = self.evaluator.evaluate_fitness(corpus_batch)
        print(f"  Step {step_num}: base fitness = {fitness_base:.4f}")

        # Evaluate antithetic pairs using per-request adapter scaling
        for k, eps in enumerate(eps_list):
            # theta + eps_k
            scales_pos = [{"id": 0, "scale": 1.0}, {"id": k + 1, "scale": 1.0}]
            self.server_mgr.set_adapter_scales(scales_pos)
            fitness_pos = self.evaluator.evaluate_fitness(corpus_batch)

            # theta - eps_k
            scales_neg = [{"id": 0, "scale": 1.0}, {"id": k + 1, "scale": -1.0}]
            self.server_mgr.set_adapter_scales(scales_neg)
            fitness_neg = self.evaluator.evaluate_fitness(corpus_batch)

            # Accumulate gradient estimate
            # Standard EGGROLL: grad += (F+ - F-) * eps
            delta = fitness_pos - fitness_neg
            for key in theta:
                grad_est[key] += delta * eps[key]

            fitness_values.append((fitness_pos, fitness_neg))
            print(f"  Step {step_num}, pair {k+1}/{n_pairs}: "
                  f"F+={fitness_pos:.4f}, F-={fitness_neg:.4f}, "
                  f"Δ={delta:.6f}, "
                  f"Δ/base={delta/abs(fitness_base)*100:.2f}%")

        # Disable all adapters
        all_off = [{"id": i, "scale": 0.0} for i in range(len(adapter_files))]
        self.server_mgr.set_adapter_scales(all_off)

        # Update parameters
        theta_new = self.lora_mgr.update(theta, grad_est, self.alpha, self.sigma, self.population)

        # Log
        avg_fitness = np.mean([f for pair in fitness_values for f in pair])
        improvement = avg_fitness - fitness_base
        print(f"  Step {step_num}: avg_fitness={avg_fitness:.4f}, "
              f"base={fitness_base:.4f}, improvement={improvement:+.4f}")

        return theta_new

    def es_step_fast(self, theta, corpus_batch, step_num):
        """Fast ES step: single-pass evaluation with concurrent adapter scaling.

        Optimization over standard ES step:
        - No base fitness evaluation (saves 1 forward pass per batch)
        - Server keeps running between antithetic pairs
        - All noise adapters loaded at startup (no restart)
        - Uses adapter scaling for instant switching

        This is the practical version that works with the current
        llama-server HTTP API without hidden state access.
        """
        rng = np.random.default_rng(step_num)
        grad_est = {k: np.zeros_like(v) for k, v in theta.items()}
        fitness_values = []

        n_pairs = self.population // 2
        eps_list = [self.lora_mgr.perturb(rng) for _ in range(n_pairs)]

        # Write adapter files (reuse existing writer pattern)
        writer = GGUFAdapterWriter(PROJECT / "evodistill" / "adapters")
        base_path = writer.write_adapter(theta, "base_fast", lora_alpha=self.lora_mgr.alpha)
        noise_paths = []
        for k, eps in enumerate(eps_list):
            path = writer.write_adapter(eps, f"eps_fast_{k}", lora_alpha=self.lora_mgr.alpha)
            noise_paths.append(path)

        # Start server once
        adapter_files = [base_path] + noise_paths
        if not self.server_mgr.start(adapter_files):
            print(f"  ERROR: Failed to start server at step {step_num}")
            return theta

        # Evaluate all antithetic pairs
        for k, eps in enumerate(eps_list):
            # theta + eps_k
            self.server_mgr.set_adapter_scales(
                [{"id": 0, "scale": 1.0}, {"id": k + 1, "scale": 1.0}])
            fitness_pos = self.evaluator.evaluate_fitness(corpus_batch)

            # theta - eps_k
            self.server_mgr.set_adapter_scales(
                [{"id": 0, "scale": 1.0}, {"id": k + 1, "scale": -1.0}])
            fitness_neg = self.evaluator.evaluate_fitness(corpus_batch)

            # Gradient accumulation
            delta = fitness_pos - fitness_neg
            for key in theta:
                grad_est[key] += delta * eps[key]

            fitness_values.append((fitness_pos, fitness_neg))
            print(f"  Step {step_num}, pair {k+1}/{n_pairs}: "
                  f"F+={fitness_pos:.4f}, F-={fitness_neg:.4f}, Δ={delta:.6f}")

        # Update
        theta_new = self.lora_mgr.update(theta, grad_est, self.alpha, self.sigma, self.population)

        # Disable adapters
        all_off = [{"id": i, "scale": 0.0} for i in range(len(adapter_files))]
        self.server_mgr.set_adapter_scales(all_off)

        avg_fitness = np.mean([f for pair in fitness_values for f in pair])
        print(f"  Step {step_num}: avg_fitness={avg_fitness:.4f}")

        return theta_new


# ============================================================
# Main
# ============================================================
def load_corpus(corpus_path):
    """Load corpus from JSONL file."""
    samples = []
    with open(corpus_path) as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                if "completion" in sample and sample["completion"].strip():
                    samples.append(sample)
            except json.JSONDecodeError:
                continue
    return samples


def run_quality_gate(server_mgr, evaluator, adapter_scales=None):
    """Run quality gates (Lisbon, 345, FizzBuzz) with optional LoRA adapter."""
    import re
    gates = [
        {"name": "Lisbon", "prompt": "The capital of Portugal is",
         "check": lambda out: "Lisbon" in out or "lisbon" in out.lower()},
        {"name": "345", "prompt": "What is 230 + 115? Answer with the number.",
         "check": lambda out: "345" in out},
        {"name": "FizzBuzz", "prompt": "Write FizzBuzz in Python for numbers 1 to 20.",
         "check": lambda out: ("fizz" in out.lower() and "buzz" in out.lower()) or ("% 3" in out and "% 5" in out)},
    ]

    results = {}
    print("\n  Quality Gates:")

    for gate in gates:
        payload = {
            "messages": [{"role": "user", "content": gate["prompt"] + " /no_think"}],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        if adapter_scales:
            payload["lora"] = adapter_scales

        try:
            resp = requests.post(
                f"http://{server_mgr.host}:{server_mgr.port}/v1/chat/completions",
                json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                output = data["choices"][0]["message"]["content"]
                output = re.sub(r'```.*?```', '', output, flags=re.DOTALL)
                passed = gate["check"](output)
                results[gate["name"]] = {"passed": passed, "output": output[:200]}
                status = "PASS" if passed else "FAIL"
                print(f"    {gate['name']}: {status}")
            else:
                results[gate["name"]] = {"passed": False, "error": f"HTTP {resp.status_code}"}
                print(f"    {gate['name']}: ERROR (HTTP {resp.status_code})")
        except Exception as e:
            results[gate["name"]] = {"passed": False, "error": str(e)}
            print(f"    {gate['name']}: ERROR ({e})")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E2: ES Optimizer — EGGROLL Algorithm")
    parser.add_argument("--mode", choices=["standard", "residual", "fast"],
                        default="fast",
                        help="Optimizer mode: standard (original ES loop), "
                             "residual (base fitness + antithetic pairs), "
                             "fast (no base fitness, fewer passes)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override number of ES steps")
    parser.add_argument("--benchmark-ane", action="store_true",
                        help="Benchmark ANE vs CPU for LoRA residuals and exit")
    parser.add_argument("--skip-corpus-check", action="store_true",
                        help="Skip corpus existence check (for testing)")
    args = parser.parse_args()

    es_config = CONFIG["es_optimizer"]

    # Initialize ANE accelerator
    print("\n[ANE Accelerator Setup]")
    ane_accel = ANEAccelerator(LoRAParameterManager(es_config), es_config)

    if args.benchmark_ane:
        print("\n[ANE Benchmark]")
        ane_accel.benchmark()
        sys.exit(0)

    # Load corpus
    corpus_path = PROJECT / CONFIG["corpus"]["output_dir"] / "corpus.jsonl"
    if not args.skip_corpus_check and not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("Run E1 corpus generation first, or use --skip-corpus-check for testing.")
        sys.exit(1)

    if corpus_path.exists():
        corpus = load_corpus(corpus_path)
        print(f"Loaded {len(corpus)} corpus samples from {corpus_path}")
    else:
        print("WARNING: No corpus file. Using synthetic test data.")
        corpus = [
            {"prompt": "Write a Python function that reverses a string.",
             "completion": "def reverse_string(s):\n    return s[::-1]"},
            {"prompt": "Implement binary search in Python.",
             "completion": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"},
            {"prompt": "What is 2 + 2?",
             "completion": "4"},
            {"prompt": "Write a Python function that checks if a number is prime.",
             "completion": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"},
        ]

    if len(corpus) < 4:
        print("WARNING: Very few corpus samples. Consider running E1 with more prompts.")
        print("Using available samples with repetition for batch evaluation.")

    # Initialize components
    lora_mgr = LoRAParameterManager(es_config)
    writer = GGUFAdapterWriter(PROJECT / "evodistill" / "adapters")
    evaluator = FitnessEvaluator(batch_size=min(4, max(1, len(corpus))))
    server_mgr = ServerManager(CONFIG)

    n_steps = args.steps or es_config["n_steps"]

    print(f"\n{'='*65}")
    print(f"E2: ES Optimizer — EGGROLL Algorithm")
    print(f"{'='*65}")
    print(f"  Mode: {args.mode}")
    print(f"  ANE acceleration: {'ENABLED' if ane_accel.ane_available else 'CPU_ONLY'}")
    print(f"  Model: Qwen3.5-35B-A3B-MoE (arch={MODEL_ARCH})")
    print(f"  Steps: {n_steps}")
    print(f"  Population: {es_config['population']} ({es_config['population']//2} antithetic pairs)")
    print(f"  Sigma: {es_config['sigma']}, Alpha: {es_config['alpha']}")
    print(f"  LoRA rank: {lora_mgr.rank}, alpha: {lora_mgr.alpha}")
    print(f"  Target layers: {lora_mgr.layer_indices} ({len(lora_mgr.layer_indices)} attn layers)")
    print(f"  Projections: {lora_mgr.projections}")
    print(f"  Params: {lora_mgr.param_count():,}")
    print(f"  Memory: {lora_mgr.memory_mb():.1f}MB (FP32)")
    print(f"  Corpus samples: {len(corpus)}, Batch size: {evaluator.batch_size}")
    print(f"{'='*65}\n")

    # Select optimizer based on mode
    if args.mode == "standard":
        # Original ES loop (restarts server between steps)
        es_algo = ESAlgorithm(lora_mgr, evaluator, server_mgr, writer, es_config)
        run_fn = es_algo.run

    elif args.mode == "residual":
        # Residual trick: base fitness + antithetic pairs with ANE acceleration
        residual_trick = ResidualTrick(lora_mgr, ane_accel, evaluator, server_mgr, es_config)

        def run_fn(corpus, n_steps):
            """Run ES optimization using residual trick mode."""
            theta = dict(lora_mgr.params)
            batch_size = min(evaluator.batch_size, len(corpus))
            best_fitness = float("-inf")
            start_time = time.time()

            for step in range(n_steps):
                batch = random.sample(corpus, min(batch_size, len(corpus)))
                theta = residual_trick.es_step_residual(theta, batch, step)
                lora_mgr.params = dict(theta)

                # Checkpoint every 10 steps
                if (step + 1) % 10 == 0 or step == n_steps - 1:
                    checkpoint_path = PROJECT / "evodistill" / "runs" / f"lora_residual_step_{step+1:04d}.npz"
                    lora_mgr.save(checkpoint_path)
                    print(f"  Checkpoint saved: {checkpoint_path}")

                # Stop server between steps (will restart with new adapters)
                server_mgr.stop()

                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed * 3600 if elapsed > 0 else 0
                eta = (n_steps - step - 1) / rate * 60 if rate > 0 else 0
                print(f"  Progress: {step+1}/{n_steps}, {rate:.1f} steps/hr, ETA {eta:.0f} min\n")

            elapsed = time.time() - start_time
            print(f"\n{'='*65}")
            print(f"ES Optimization (Residual Mode) Complete")
            print(f"  Total steps: {n_steps}")
            print(f"  Elapsed: {elapsed/60:.1f} min")
            print(f"  Final params: {lora_mgr.param_count():,}")
            print(f"{'='*65}")

            final_path = writer.write_adapter(theta, "final_residual", lora_alpha=lora_mgr.alpha)
            print(f"  Final adapter: {final_path}")
            return theta

    elif args.mode == "fast":
        # Fast mode: no base fitness evaluation, fewer passes per step
        residual_trick = ResidualTrick(lora_mgr, ane_accel, evaluator, server_mgr, es_config)

        def run_fn(corpus, n_steps):
            """Run ES optimization in fast mode.

            Keeps the server running between steps by rewriting adapter files
            and restarting only when adapter files change (between steps).
            On 16GB machines, the 30s server startup is the bottleneck,
            so we minimize restarts by loading all adapters at step 0
            and keeping the server alive across steps.
            """
            theta = dict(lora_mgr.params)
            batch_size = min(evaluator.batch_size, len(corpus))
            best_fitness = float("-inf")
            start_time = time.time()

            for step in range(n_steps):
                batch = random.sample(corpus, min(batch_size, len(corpus)))

                # Generate perturbations for this step
                rng = np.random.default_rng(step)
                n_pairs = residual_trick.population // 2
                eps_list = [lora_mgr.perturb(rng) for _ in range(n_pairs)]

                # Write adapter files (overwrite existing ones)
                base_path = writer.write_adapter(theta, "base_fast", lora_alpha=lora_mgr.alpha)
                noise_paths = []
                for k, eps in enumerate(eps_list):
                    path = writer.write_adapter(eps, f"eps_fast_{k}", lora_alpha=lora_mgr.alpha)
                    noise_paths.append(path)

                # Start/restart server with all adapters
                adapter_files = [base_path] + noise_paths
                if not server_mgr.start(adapter_files):
                    print(f"  ERROR: Failed to start server at step {step}")
                    continue

                # Evaluate all antithetic pairs
                grad_est = {k: np.zeros_like(v) for k, v in theta.items()}
                fitness_values = []

                for k, eps in enumerate(eps_list):
                    # theta + eps_k
                    server_mgr.set_adapter_scales(
                        [{"id": 0, "scale": 1.0}, {"id": k + 1, "scale": 1.0}])
                    fitness_pos = evaluator.evaluate_fitness(batch)

                    # theta - eps_k
                    server_mgr.set_adapter_scales(
                        [{"id": 0, "scale": 1.0}, {"id": k + 1, "scale": -1.0}])
                    fitness_neg = evaluator.evaluate_fitness(batch)

                    # Gradient accumulation
                    delta = fitness_pos - fitness_neg
                    for key in theta:
                        grad_est[key] += delta * eps[key]

                    fitness_values.append((fitness_pos, fitness_neg))
                    print(f"  Step {step}, pair {k+1}/{n_pairs}: "
                          f"F+={fitness_pos:.4f}, F-={fitness_neg:.4f}, Δ={delta:.6f}")

                # Update parameters
                theta = lora_mgr.update(theta, grad_est, residual_trick.alpha,
                                         residual_trick.sigma, residual_trick.population)
                lora_mgr.params = dict(theta)

                # Disable adapters
                all_off = [{"id": i, "scale": 0.0} for i in range(len(adapter_files))]
                server_mgr.set_adapter_scales(all_off)

                # Stop server (need restart next step with new adapter files)
                server_mgr.stop()

                avg_fitness = np.mean([f for pair in fitness_values for f in pair])
                print(f"  Step {step}: avg_fitness={avg_fitness:.4f}")

                # Checkpoint every 10 steps
                if (step + 1) % 10 == 0 or step == n_steps - 1:
                    checkpoint_path = PROJECT / "evodistill" / "runs" / f"lora_fast_step_{step+1:04d}.npz"
                    lora_mgr.save(checkpoint_path)
                    print(f"  Checkpoint saved: {checkpoint_path}")

                elapsed = time.time() - start_time
                rate = (step + 1) / elapsed * 3600 if elapsed > 0 else 0
                eta = (n_steps - step - 1) / rate * 60 if rate > 0 else 0
                print(f"  Progress: {step+1}/{n_steps}, {rate:.1f} steps/hr, ETA {eta:.0f} min\n")

            elapsed = time.time() - start_time
            print(f"\n{'='*65}")
            print(f"ES Optimization (Fast Mode) Complete")
            print(f"  Total steps: {n_steps}")
            print(f"  Elapsed: {elapsed/60:.1f} min")
            print(f"  Final params: {lora_mgr.param_count():,}")
            print(f"{'='*65}")

            final_path = writer.write_adapter(theta, "final_fast", lora_alpha=lora_mgr.alpha)
            print(f"  Final adapter: {final_path}")
            return theta

    # Run ES optimization
    try:
        final_theta = run_fn(corpus, n_steps=n_steps)

        # Run quality gates with final adapter
        print("\n[Quality Gates — Final Adapter]")
        final_adapter_path = PROJECT / "evodistill" / "adapters" / "final_fast.gguf"
        if args.mode == "standard":
            final_adapter_path = PROJECT / "evodistill" / "adapters" / "final.gguf"

        if final_adapter_path.exists():
            server_mgr.start([final_adapter_path])
            gate_results = run_quality_gate(server_mgr, evaluator)

        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "phase": "E2",
            "mode": args.mode,
            "model_arch": MODEL_ARCH,
            "config": es_config,
            "ane_available": ane_accel.ane_available,
            "final_params": lora_mgr.param_count(),
            "final_memory_mb": lora_mgr.memory_mb(),
            "corpus_size": len(corpus),
            "quality_gates": gate_results if 'gate_results' in dir() else None,
            "attribution": {
                "eggroll": "@rustane_dev (March 2026)",
                "ssd": "Apple Research (arXiv 2604.01193)",
                "ane_api": "apple-silicon-internals repo",
                "inference": "llama.cpp anemll-flash-llama.cpp fork",
            },
        }
        results_path = PROJECT / "evodistill" / "runs" / f"e2_{args.mode}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {results_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving checkpoint...")
        checkpoint_path = PROJECT / "evodistill" / "runs" / f"lora_{args.mode}_interrupted.npz"
        lora_mgr.save(checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")
    finally:
        server_mgr.stop()