"""Minimal integration test: load draft weights, run a dummy forward pass."""

import time

import mlx.core as mx

from draft_forward import draft_forward
from loader import load_draft_weights

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS_PATH = (
    "/Users/manuelmonteiro/.cache/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash"
    "/snapshots/a6ab3a277f856d91c43f28711611e7929073d56d/model.safetensors"
)
HIDDEN_SIZE = 2048
BLOCK_SIZE = 16
BATCH = 1


def main() -> None:
    print("1. Loading draft weights ...")
    t0 = time.time()
    weights = load_draft_weights(WEIGHTS_PATH)
    print(f"   Loaded {len(weights)} tensors in {time.time()-t0:.2f}s")

    # Verify expected keys exist
    required = [
        "norm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.7.self_attn.o_proj.weight",
        "layers.7.mlp.down_proj.weight",
    ]
    for k in required:
        assert k in weights, f"Missing weight: {k}"
    print("   Required keys present.")

    print(f"\n2. Creating dummy hidden states: shape ({BATCH}, {BLOCK_SIZE}, {HIDDEN_SIZE})")
    # Random normal inits for the block latents — in a real hybrid loop these
    # come from the target embedding service.
    hidden_states = mx.random.normal(shape=(BATCH, BLOCK_SIZE, HIDDEN_SIZE)).astype(mx.bfloat16)
    mx.eval(hidden_states)  # materialise before timing
    print(f"   dtype={hidden_states.dtype}")

    print("\n3. Running draft_forward(hidden_states, weights) ...")
    t1 = time.time()
    output = draft_forward(hidden_states, weights)
    mx.eval(output)  # force synchronous completion
    elapsed = time.time() - t1
    print(f"   Done in {elapsed:.3f}s")

    assert output.shape == hidden_states.shape, (
        f"Output shape mismatch: {output.shape} != {hidden_states.shape}"
    )
    print(f"   Output shape: {output.shape}")
    print(f"   Output dtype: {output.dtype}")
    print(f"   Output sample (first 5 values pos 0): {output[0, 0, :5]}")

    print("\n4. Running a second pass for warm-up timing ...")
    t2 = time.time()
    output2 = draft_forward(hidden_states, weights)
    mx.eval(output2)
    elapsed2 = time.time() - t2
    print(f"   Second pass: {elapsed2:.3f}s")

    print("\n✅ All tests passed — draft forward runs without errors.")


if __name__ == "__main__":
    main()
