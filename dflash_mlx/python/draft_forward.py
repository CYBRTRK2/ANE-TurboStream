"""Minimal MLX-based draft transformer forward (P0 — self-attention only).

Architecture (from config.json + architecture_design.md):
- 8 draft layers, hidden_size=2048
- 32 attention heads, 4 KV heads, head_dim=128
- MLP intermediate_size=6144 (gate, up, down proj)
- RMSNorm (eps=1e-6)
- RoPE base=10_000_000, traditional=False
- NO cross-attention (P1+)
- NO lm_head / token_embd — target provides those externally.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
from mlx_lm.models.rope_utils import initialize_rope

# ---------------------------------------------------------------------------
# Architecture constants (from config.json)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 2048
NUM_LAYERS = 8
N_HEADS = 32
N_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE_SIZE = 6144
RMS_NORM_EPS = 1e-6
ROPE_BASE = 10_000_000
MAX_POS = 262_144
SCALE = HEAD_DIM ** -0.5


def _rms_norm(x: mx.array, weight: mx.array, eps: float = RMS_NORM_EPS) -> mx.array:
    """Apply RMSNorm manually (MLX nn.RMSNorm equivalent).
    x:     any shape, last dim == weight size
    weight: 1-D array [dim]
    """
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def _linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    """x @ W.T  — weight stored (out, in) like nn.Linear."""
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def _mlp(
    x: mx.array,
    gate_w: mx.array,
    up_w: mx.array,
    down_w: mx.array,
) -> mx.array:
    """SiLU-gated MLP: down( silu(gate(x)) * up(x) )."""
    gate = _linear(x, gate_w)
    up = _linear(x, up_w)
    hidden = mx.sigmoid(gate) * gate * up  # SiLU = sigmoid(x) * x
    return _linear(hidden, down_w)


def draft_forward(
    hidden_states: mx.array,
    weights: dict[str, mx.array],
) -> mx.array:
    """Run the 8-layer draft model.

    Args
    ----
    hidden_states: [batch, seq_len, hidden_size] — already embedded.
                   For the P0 test: (1, 16, 2048).
    weights: dict loaded by ``loader.load_draft_weights``.

    Returns
    -------
    Final hidden states: [batch, seq_len, hidden_size].
    """
    batch, seq_len, dim = hidden_states.shape
    if dim != HIDDEN_SIZE:
        raise ValueError(f"Expected hidden_size={HIDDEN_SIZE}, got {dim}")

    # Shared RoPE (offset 0 because we do one-shot block processing)
    rope = initialize_rope(
        HEAD_DIM,
        base=ROPE_BASE,
        traditional=False,
        scaling_config=None,
        max_position_embeddings=MAX_POS,
    )

    for layer_idx in range(NUM_LAYERS):
        pfx = f"layers.{layer_idx}"

        # ---- pre-attention norm ----
        residual = hidden_states
        hidden_states = _rms_norm(
            hidden_states, weights[f"{pfx}.input_layernorm.weight"]
        )

        # ---- self-attention projections ----
        q = _linear(hidden_states, weights[f"{pfx}.self_attn.q_proj.weight"])
        k = _linear(hidden_states, weights[f"{pfx}.self_attn.k_proj.weight"])
        v = _linear(hidden_states, weights[f"{pfx}.self_attn.v_proj.weight"])

        # reshape: (batch, seq, n_heads, head_dim)
        q = q.reshape(batch, seq_len, N_HEADS, HEAD_DIM)
        k = k.reshape(batch, seq_len, N_KV_HEADS, HEAD_DIM)
        v = v.reshape(batch, seq_len, N_KV_HEADS, HEAD_DIM)

        # per-head q/k RMSNorm
        q = _rms_norm(q, weights[f"{pfx}.self_attn.q_norm.weight"])
        k = _rms_norm(k, weights[f"{pfx}.self_attn.k_norm.weight"])

        # transpose -> (batch, n_heads, seq, head_dim)
        queries = q.transpose(0, 2, 1, 3)
        keys = k.transpose(0, 2, 1, 3)
        values = v.transpose(0, 2, 1, 3)

        # RoPE
        queries = rope(queries, offset=0)
        keys = rope(keys, offset=0)

        # Scaled dot-product attention — causal mask for the 16-token block
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=SCALE, mask="causal"
        )

        # (batch, n_heads, seq, head_dim) -> (batch, seq, n_heads*head_dim)
        output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

        # out projection + residual
        hidden_states = residual + _linear(
            output, weights[f"{pfx}.self_attn.o_proj.weight"]
        )

        # ---- post-attention norm + MLP ----
        residual = hidden_states
        hidden_states = _rms_norm(
            hidden_states, weights[f"{pfx}.post_attention_layernorm.weight"]
        )
        hidden_states = residual + _mlp(
            hidden_states,
            gate_w=weights[f"{pfx}.mlp.gate_proj.weight"],
            up_w=weights[f"{pfx}.mlp.up_proj.weight"],
            down_w=weights[f"{pfx}.mlp.down_proj.weight"],
        )

    # ---- final norm ----
    hidden_states = _rms_norm(hidden_states, weights["norm.weight"])
    return hidden_states
