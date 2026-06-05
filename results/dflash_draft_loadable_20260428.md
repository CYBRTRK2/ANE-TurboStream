# dFlash Draft Loadable GGUF — 2026-04-28

Goal: remove the blocker where `--dflash-draft` could not load any DFlash draft GGUF in llama.cpp.

## Final Artifact

`/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-Draft-f16-loadable-ar-f32norm.gguf`

- Size: `1.4G`
- Tensors: standard Qwen3 8-layer draft tensors plus raw-copied target `token_embd.weight` and `output.weight`
- Custom DFlash tensors omitted for standard llama.cpp loading:
  - `fc.weight`
  - `hidden_norm.weight`

## Converter Fixes

`scripts/convert_dflash_draft_to_gguf.py` now:

- writes real llama.cpp metadata for `attention.key_length`, `attention.value_length`, and `rope.dimension_count`;
- preserves MLX/PyTorch `[out, in]` arrays so GGUFWriter emits llama.cpp's expected GGML dimensions;
- skips custom DFlash projection tensors by default because standard llama.cpp rejects unused tensors;
- keeps 1D/norm tensors in f32 so CPU draft decode does not abort on f32/f16 binary ops.

`scripts/build_loadable_dflash_draft.py` adds:

- raw-copy append of target `token_embd.weight`;
- raw-copy append of target `output.weight`;
- no overwrite unless `DFLASH_OVERWRITE=1`.

## Verification

Command shape:

```bash
env GGML_COREML_ENABLE=0 ./build-nsg-opt/bin/dflash-cli \
  -m /Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --dflash \
  --dflash-draft /Users/manuelmonteiro/models/Qwen3.5-35B-A3B-Draft-f16-loadable-ar-f32norm.gguf \
  --draft-ngl 0 --moe-topk 4 -ngl 99 -c 512 \
  -n 4 --block-size 4 --max-cycles 1 \
  -p "The capital of Portugal is"
```

Result:

- Draft model loads successfully.
- Draft context is prefilled with the prompt.
- One-cycle probe runs to completion.
- Acceptance remains `0/3`.
- CPU draft time for 3 draft tokens was `6584.54 ms`; this is correctness-only, not a performance path.

## Interpretation

The old hard blocker "draft GGUF cannot load" is fixed for AR fallback experiments.

This does not make dFlash fast yet. The useful next step is either:

- run draft on GPU/ANE without evicting the target, or
- implement the real block-diffusion draft runtime/projector path instead of CPU autoregressive fallback.

