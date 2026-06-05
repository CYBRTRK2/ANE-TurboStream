# ANE-TurboStream v3 Physical Ceiling — Running Summary

**Updated:** 2026-04-28 00:12 WEST  
**Workspace:** `/Users/manuelmonteiro/Desktop/ANE project`  
**Plan:** `docs/plans/2026-04-25-ane-turbostream-v3-physical-ceiling_NEWPLAN.md`

## Current State

Official baseline remains Apr 25: topk=4 is `26.02 +/- 0.25 t/s` from `results/baseline_20260425/summary.md`. A live Apr 27 re-anchor under current load measured about `21.8 t/s`, so current AutoResearch uses live controls instead of historical fixed thresholds.

## Changes Landed In This Pass

- CoreML scaffold is now opt-in with `GGML_COREML_ENABLE=1`; default benchmarks no longer expose an ANE device from an unsupported backend.
- `scripts/bench_protocol.sh` now uses valid `llama-bench --output json` flags, stock+shared-only flags, configurable run counts, and explicit CoreML-disable defaults.
- `scripts/autoresearch_loop.py` no longer executes on import, parses current llama-bench JSON correctly, measures a live control row, and supports bounded smoke runs.
- DFlash C++ block path fixed staged-token duplication, accepted-position logits, rollback/recompute safety, and added `--max-cycles` for one-cycle acceptance probes.
- DFlash stats now report `accepted/drafted` draft tokens instead of `accepted/generated`.
- `llama-lookup` now handles recurrent/M-RoPE cache crop failures by recomputing committed state, fails on bad verify decodes, and adaptively disables n-gram drafting after repeated low-acceptance recomputes.
- Track 1 lookup result: repeated/context-cache prompt hit `79.807 t/s` with `64/64` accepted; normal prompt fell to `12.752 t/s` with `8/64` accepted and is not a daily-driver win.
- `llama-lookahead` now exposes W/N/G env controls and fails cleanly, but remains blocked by M-RoPE auxiliary sequence state after the first coupled-sequence splitter fix.
- Converted DFlash draft loading is now fixed for C++ AR fallback experiments. Final artifact: `/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-Draft-f16-loadable-ar-f32norm.gguf`.
- The loadable draft now reaches a real one-cycle `dflash-cli --dflash-draft ... --max-cycles 1` probe, but acceptance remains `0/3` and CPU draft time was `6584.54 ms` for three drafted tokens.
- Track 11A profiling scaffold was corrected: `GGML_METAL_PROFILE=1` now records per-op CPU encode timing, classifies the LM head on the correct GGML dimension, and no longer labels encode overhead as GPU timing.
- MLX DFlash reference code is importable, but a no-download acceptance proof is blocked by missing complete local MLX target/draft artifacts.
- AutoResearch one-rep sweep found transient `threads=6/8` candidates; 3-run validation plus immediate `threads=4` rerun rejected them as noise. Keep `threads=4`.
- Track 11A MPSGraphExecutable map is ready: first targets are shared expert dense block or full-attention projections; main risk is MPSGraph command-buffer integration with existing Metal compute encoders.

## Verified Commands

- `cmake --build build-nsg-opt --target ggml-coreml -j 4`
- `cmake --build build-nsg-opt --target dflash-cli -j 4`
- `cmake --build build-nsg-opt --target llama-lookup -j 4`
- `cmake --build build-nsg-opt --target llama-lookahead -j 4`
- `python3 -m py_compile scripts/autoresearch_loop.py`
- `bash -n scripts/bench_protocol.sh`
- `env AUTORESEARCH_MAX=2 AUTORESEARCH_REPETITIONS=1 GGML_COREML_ENABLE=0 python3 scripts/autoresearch_loop.py`
- `env AUTORESEARCH_REPETITIONS=1 GGML_COREML_ENABLE=0 python3 scripts/autoresearch_loop.py`
- `./build-nsg-opt/bin/dflash-cli ... --dflash --block-size 16 --max-cycles 1`
- `./build-nsg-opt/bin/llama-lookup ... --draft 16 --draft-min 0`
- `LLAMA_LOOKAHEAD_W=1 LLAMA_LOOKAHEAD_G=1 LLAMA_LOOKAHEAD_N=2 ./build-nsg-opt/bin/llama-lookahead ...`
- `./build-nsg-opt/bin/llama-bench ... -t {4,6,8} -r 3 --moe-topk 4 --output json`

## Next Best Actions

1. Run the full baseline protocol when the machine is cool/quiet.
2. Continue Track 3 by acquiring/building a complete MLX target+draft pair or implementing the custom C++ DFlash draft runtime.
3. Continue Track 2 by implementing real CoreML MIL support behind the opt-in gate.
4. Start Track 11A only behind an env flag and a sidecar MPSGraph command-buffer path.
5. Only revisit lookahead after designing recurrent/M-RoPE-safe auxiliary sequence cleanup.
