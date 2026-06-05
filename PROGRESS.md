# ANE-TurboStream v3 Physical Ceiling — Progress Tracker

**Last updated:** 2026-04-28 00:12 WEST  
**Active plan:** `docs/plans/2026-04-25-ane-turbostream-v3-physical-ceiling_NEWPLAN.md`  
**Target model:** `/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf`

## Current Official Baseline

Official Apr 25 baseline remains the reference until a full protocol rerun completes:

| Config | Decode t/s | Source |
|---|---:|---|
| stock topk=8 | 23.77 +/- 0.42 | `results/baseline_20260425/summary.md` |
| stock topk=4 | 26.02 +/- 0.25 | `results/baseline_20260425/summary.md` |
| shared-only | 32.89 +/- 0.53 | `results/baseline_20260425/summary.md` |

Live Apr 27 re-anchor during this session was lower (`topk=4 tg128 ~= 21.8 t/s`) under current system/thermal load. Treat it as a live control for AutoResearch, not as a replacement official baseline.

## Completed This Session

- Fixed CoreML scaffold leakage: `libggml-coreml` remains linkable, but the ANE device is hidden unless `GGML_COREML_ENABLE=1`.
- Verified default devices are now `MTL0, BLAS`; `GGML_COREML_ENABLE=1` exposes `ANE`.
- Removed md5-verified duplicate `libggml 2/3` symlinks from `build-nsg-opt/bin/`.
- Fixed `scripts/bench_protocol.sh` for valid `llama-bench` flags and explicit `GGML_COREML_ENABLE=0` on non-ANE baselines.
- Fixed `scripts/autoresearch_loop.py`: import no longer runs the loop, JSON parsing uses `avg_ts`, and the loop measures a live control before judging hypotheses.
- AutoResearch smoke passed: live control `21.79 t/s`; threads 2 and 6 discarded against that live control.
- AutoResearch one-rep sweep found transient thread candidates, but 3-run validation rejected them; keep `threads=4` as default.
- Hardened DFlash C++ generation:
  - no duplicate staged token,
  - next-token logits come from the last accepted position,
  - partial rollback attempts `llama_memory_seq_rm`,
  - recurrent/GDN rollback falls back to full committed-prefix recompute,
  - `--max-cycles` supports cheap one-cycle acceptance probes,
  - stats now report accepted draft tokens over drafted tokens instead of generated tokens.
- Fixed `llama-lookup` correctness on recurrent/M-RoPE failures:
  - rejected-draft cache crop failures now recompute the committed prefix,
  - verify decode failures are fatal instead of silently counted,
  - low-acceptance repeated recomputes disable n-gram drafting adaptively.
- Probed Track 1 speculative alternatives; details in `results/track1_speculative_20260427.md`.
- Added partial `llama-lookahead` hardening:
  - W/N/G can be controlled with `LLAMA_LOOKAHEAD_W`, `LLAMA_LOOKAHEAD_N`, `LLAMA_LOOKAHEAD_G`,
  - prompt decode failures are checked,
  - coupled-sequence batches can use non-sequential equal splitting in memory splitters.
- Confirmed MLX DFlash reference code imports, but local MLX acceptance proof is blocked by missing target/draft model artifacts.
- Fixed the loadable-draft blocker for C++ DFlash AR fallback experiments:
  - generated `/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-Draft-f16-loadable-ar-f32norm.gguf`,
  - added raw-copied target `token_embd.weight` and `output.weight`,
  - corrected draft GGUF metadata/shapes/norm tensor dtypes,
  - verified `dflash-cli --dflash-draft ... --max-cycles 1` loads both target and draft, prefills both contexts, and runs a one-cycle probe.
- Fixed the Track 11A Metal profiling scaffold:
  - `GGML_METAL_PROFILE=1` now records per-op encode timing instead of only starting a dead timer,
  - LM-head classification now uses the correct GGML dimension,
  - output is labeled CPU encode timing, not GPU timing.

## dFlash State

`dflash-cli --dflash --max-cycles 1` now runs a real one-cycle verify probe. Placeholder copy-token drafting still gives `0/3` acceptance in the small smoke and `0/15` in block-size-16 probes, as expected.

The C++ AR fallback draft-load blocker is fixed. The loadable artifact is:

`/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-Draft-f16-loadable-ar-f32norm.gguf`

Verification on 2026-04-28:

- draft model loads successfully,
- draft context is prefilled,
- one-cycle DFlash probe completes,
- acceptance remains `0/3`,
- CPU draft time for three drafted tokens was `6584.54 ms`.

Conclusion: the old loader/model-format blocker is gone for AR fallback experiments, but this is not a performance path yet. The next real dFlash work is GPU/ANE draft execution or the actual block-diffusion projector/runtime path.

## Track 1 Speculative Alternatives

`llama-lookup` is now honest on this model:

- repeated/cacheable prompt: `79.807 t/s`, `64/64` draft tokens accepted, zero recomputes;
- normal explanatory prompt: `12.752 t/s`, `8/64` accepted, four recurrent-state recomputes, adaptive drafting disabled.

Conclusion: lookup is useful for repeated/context-cache workloads, not as a general daily-driver path.

`llama-lookahead` remains blocked:

- default W=15/G=15 OOMs with 31 recurrent sequences;
- tiny W=1/G=1/N=2 no longer dies at the first coupled-sequence split, but still fails on stale M-RoPE auxiliary-sequence positions.

Conclusion: lookahead needs a recurrent/M-RoPE-safe auxiliary sequence cleanup or recompute strategy before benchmarking.

## Track 3 MLX Proof State

`dflash-mlx-main` is present and importable, and `dflash-benchmark --help` works with local `mlx 0.31.1`.

No no-download MLX acceptance proof is feasible with current local artifacts:

- 9B target/draft MLX pair is not local.
- 35B DFlash MLX draft is local, but complete 35B MLX target is missing.
- GGUF target/draft files do not satisfy the stock MLX DFlash runtime.

## Track 11A MPSGraphExecutable Map

Track 11A should remain independent of CoreML/ANE and be guarded by a GPU-side env flag.

Best first targets:

- shared expert dense block (`ffn_up_shexp`, `ffn_gate_shexp`, SiLU/SWIGLU, `ffn_down_shexp`) if hot weights can be pre-dequantized/cached FP16;
- full-attention projections (`wq`, `wk`, `wv`, `wo`) as the simplest static-shape fallback target.

Avoid first:

- routed experts / `MUL_MAT_ID`, router top-k/argsort, KV/attention kernels, LM head.

Main integration risk:

- current Metal ops run under a live `MTLComputeCommandEncoder`; `MPSGraphExecutable` encodes to a command buffer, so Track 11A needs a sidecar command-buffer path or explicit encoder close/reopen boundaries.

## Open Tracks

| Track | Status | Next concrete step |
|---|---|---|
| Track 0 | Partial | Run full `scripts/bench_protocol.sh` when thermals/load are stable |
| Track 1 | Partially closed | `llama-lookup` niche win documented; `llama-lookahead` blocked on recurrent/M-RoPE sequence cleanup |
| Track A / 3 | C++ safer, AR draft load fixed, quality/runtime blocked | Implement real block-diffusion draft/projector path or GPU/ANE draft execution; CPU AR fallback accepts `0/3` and is far too slow |
| Track 2 | Scaffold gated | Implement real CoreML MIL graph support behind `GGML_COREML_ENABLE=1` |
| Track 5 | Fixed + swept | One-rep sweep + 3-run thread validation found no confirmed daily-driver win |
| Track 11A | Mapped | Start with shared expert dense block or attention projections; requires MPSGraph command-buffer integration |

## Track 11A Diagnostic Update

`GGML_METAL_PROFILE=1` is now usable for encode-overhead checks and op census, but it is not a GPU hot-op profiler. Do not use its percentages to justify an MPSGraph target. Use it together with `GGML_METAL_PER_TOKEN_TIMING=2` op maps, Metal capture, or a dedicated sidecar timing path.

## Key Caveat

Do not quote Apr 27 live `21.8 t/s` as a regression against Apr 25 official baseline. It was measured under a noisy live session and is only useful as an in-window control.
