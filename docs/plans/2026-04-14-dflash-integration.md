# DFlash Integration Plan for ANE-TurboStream

## Context

- **Hardware**: M4 MacBook Air 16GB, 120 GB/s unified memory
- **Target model**: Qwen3.5-35B-A3B (IQ2_M, 10.6 GB GGUF on llama.cpp)
- **Current baseline**: 26 tok/s (memory-bandwidth-bound)
- **DFlash opportunity**: 1.35-1.74x speedup via speculative decoding

## Why DFlash?

The 26 tok/s ceiling is memory-bandwidth-bound. The only ways to break through:
1. Reduce per-token memory reads (fewer experts, lower quant)
2. Get more useful output per memory read — **this is what speculative decoding does**

DFlash generates 16 draft tokens in one pass, target verifies in one pass.
With 89% acceptance on Qwen3.5-35B-A3B, we get ~9 accepted tokens per verify.
Each verify pass reads the same ~1.5 GB from memory, but produces 9x more output.
Effective: 26 * 1.5-1.7x = ~39-44 tok/s.

## Architecture: DFlash over llama.cpp

### Components needed:

1. **Target model**: Our existing GGUF/Qwen3.5-35B-A3B on anemll-flash-llama.cpp (already works)
2. **Draft model**: DFlash draft model (905MB bf16, 8-layer transformer)
   - Load via separate llama.cpp context or as a 2nd GGUF model
   - Uses target's embed_tokens and lm_head (shared weights)
3. **Speculative decode loop**: Draft-verify-accept cycle in C++
4. **Rollback for GatedDeltaNet**: Custom Metal kernel for tape-replay on linear_attn layers

### Implementation phases:

#### Phase 6A: DFlash Draft Model Integration (llama.cpp)

1. Convert DFlash draft model from HuggingFace safetensors to GGUF
2. Add speculative decoding mode to anemll-flash-llama.cpp:
   - `--moe-mode stock --speculative-draft <draft.gguf> --speculative-block-size 16`
3. Implement the draft-verify-accept loop:
   - Load draft model into separate llama context
   - Draft: run draft model on noise tokens + target hidden states
   - Verify: run target model on draft block
   - Accept: greedy match, commit accepted tokens, rollback on rejection
4. Qwen3.5-35B-A3B is hybrid (GDN + attention):
   - GDN layers need RecurrentRollbackCache equivalent
   - Attention layers need standard KV cache rollback (already in llama.cpp speculative)

#### Phase 6B: Hidden State Extraction

DFlash needs target model hidden states at layers [1, 10, 19, 28, 37].
- Modify target forward pass to capture intermediate activations at these layers
- Project concatenated hidden states through draft model's fc layer
- This requires hooking into llama.cpp's layer-by-layer computation

#### Phase 6C: Rollback for GatedDeltaNet

Qwen3.5-35B-A3B has 30 GDN layers (every 3 of 4).
Each GDN layer maintains a recurrent state that gets updated during verify.
On partial rejection, we need to rollback those states.

Options:
A. Full checkpoint/restore (expensive: states are [B, Hv, Dv, Dk])
B. Tape-replay rollback (DFlash's approach): record innovation deltas, replay only accepted steps
C. Simple approach: re-run target from last accepted position (wasteful but correct)

For MVP, start with option C (re-run), optimize to B later.

#### Phase 6D: Benchmark & Validate

- Run identical prompt on baseline vs speculative
- Verify output quality (greedy match to target model)
- Measure tok/s including draft time, verify time, and acceptance rate
- Compare against the 26 tok/s baseline

### Memory budget (16 GB):

| Component | Size |
|-----------|------|
| Target GGUF (IQ2_M) | 10.6 GB |
| DFlash draft model (4-bit GGUF) | ~0.6 GB |
| Target KV cache (q4_0, 2048 ctx) | ~0.5 GB |
| Draft KV cache | ~0.1 GB |
| System + overhead | ~2.0 GB |
| **Total** | ~13.8 GB |

Fits in 16 GB with ~2.2 GB headroom.

### Shortcut: MLX DFlash + llama.cpp target (Hybrid approach)

If C++ port is too complex initially, we can run:
- Target model on llama.cpp (our fast, stable baseline)
- Draft model on MLX (dflash-mlx code already works)
- Communication via shared memory / subprocess

This avoids porting the draft model to GGUF and gets us benchmark numbers faster.

The downside: two separate frameworks, higher memory usage (MLX Python + llama.cpp C++),
and IPC overhead between processes.

### Expected outcomes:

| Scenario | tok/s | Acceptance | Notes |
|----------|-------|------------|-------|
| Baseline (stock) | 26 | N/A | Current |
| DFlash (1024 tok) | ~42-45 | ~89% | 1.7x speedup |
| DFlash (4096 tok) | ~35-37 | ~87% | 1.35x speedup |

These are conservative estimates. M4 Air has less bandwidth than M5 Max (120 vs 400 GB/s),
but the relative speedup should be similar because:
- Acceptance rate depends on draft quality, not hardware
- Draft model inference cost is proportional to compute, which scales with bandwidth
- The ratio of draft cost to verify cost should be similar across hardware

### ANE Opportunity (Phase 6E):

DFlash's verify pass processes 16 tokens simultaneously.
This is BATCHED GEMM, not single-vector GEMV.
ANE is designed for batched operations (conv1x1 is batched matmul).
If verify takes ~34ms * batch_factor, ANE could offload part of it.

This would be Phase 4 revisited: ANE for batched DFlash verification,
NOT for per-token decode GEMV. Different problem entirely.