"""Load draft model safetensors into MLX arrays."""

import mlx.core as mx


def load_draft_weights(path: str) -> dict[str, mx.array]:
    """Load a safetensors file containing the DFlash draft weights.

    Returns a dict mapping weight name -> MLX array.
    The draft weights include 8 transformer layers, plus projector weights
    (fc.weight, hidden_norm.weight) that are tracked but not used by the
    basic draft forward.
    """
    weights = mx.load(path)
    # Ensure string keys
    return {str(k): v for k, v in weights.items()}
