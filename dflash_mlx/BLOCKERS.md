# P0 Implementation Notes / Blockers

## What was implemented

1. **loader.py** — loads the DFlash draft safetensors file into an MLX dict (bfloat16).
2. **draft_forward.py** — minimal 8-layer transformer forward:
   - Self-attention with 32 heads / 4 KV heads / head_dim=128
   - Per-head q/k RMSNorm
   - RoPE (base=10M, traditional=False)
   - SiLU-gated MLP (intermediate=6144)
   - Causal attention mask for the 16-token block
   - Returns final hidden states [batch, seq, hidden_size]
3. **test_draft.py** — integration test:
   - loads weights, creates dummy (1, 16, 2048) hidden states in bfloat16
   - runs forward, asserts shape/dtype, prints timing
   - Second pass ≈ 50 ms on this machine (MLX Metal backend)

## Deliberate P0 simplifications (per task spec)

- **NO cross-attention** — target hidden taps + prefix K/V are P1.
- **NO output_norm / lm_head** — draft does not own vocabulary head; target provides logits.
- **NO token embedding** — input is already embedded hidden states.
- **NO GDN scheduler / tape replay / block diffusion steps** — single forward pass only.
- **No KV-cache** — one-shot block processing (seq_len == block_size).

## Issues encountered

| # | Issue | Resolution |
|---|-------|------------|
| 1 | `mx.load()` on safetensors returned bfloat16 but `mx.random.normal` defaults to float32, so the forward up-cast everything to float32 silently. | Explicit `.astype(mx.bfloat16)` on dummy input to match weights; output stays bfloat16. |
| 2 | `mx.core.array` has no `.device` attribute in current MLX (0.31.1). | Removed that print from test script. |
| 3 | `initialize_rope` from `mlx_lm.models.rope_utils` is the documented/standard way to get a callable rope in the MLX-LM stack; works correctly. | No block. |

## Files created

- `dflash_mlx/python/loader.py`
- `dflash_mlx/python/draft_forward.py`
- `dflash_mlx/python/test_draft.py`
- `dflash_mlx/BLOCKERS.md` (this file)
