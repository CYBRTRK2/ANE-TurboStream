# Track 11A Metal Profile Probe — 2026-04-28

Goal: make the local Metal profiling scaffold usable before choosing a first `MPSGraphExecutable` target.

## Bug Found

`GGML_METAL_PROFILE=1` was misleading and mostly inert:

- `ggml_metal_op_encode_impl()` captured `t_op_start`, but never called `ggml_metal_profile_record()`.
- `ggml_metal_profile_graph_done()` was never reached from the graph-completion hook.
- LM-head detection checked `src0->ne[0] >= 100000`, but GGML stores the output/vocab dimension for these weights in `src0->ne[1]`.
- Special MUL_MAT categories returned before adding their time to the total, so any future percentage denominator would have been wrong.
- The log called this "GPU" timing, but the measurement is CPU encode time around Metal command encoding.

## Fix

`ggml/src/ggml-metal/ggml-metal-ops.cpp` now:

- records per-op encode elapsed time after dispatch;
- calls the profile report hook once a graph finishes;
- classifies LM head using `src0->ne[1]`;
- includes special categories in the total denominator;
- labels output as CPU encode timing rather than GPU timing.

## Smoke

Command:

```bash
env GGML_COREML_ENABLE=0 GGML_METAL_PROFILE=1 ./build-nsg-opt/bin/llama-bench \
  -m /Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  -p 0 -n 16 -r 1 -t 4 -ngl 99 --moe-topk 4 --output json
```

Result:

- benchmark completed;
- profile report emitted after 10 profiled decode graphs;
- `MUL_MAT(lm_head)` was separated correctly;
- decode sample: `25.546684 t/s`.

Important caveat: this is not a GPU hot-op profiler. It is an encode-overhead profiler and graph-op census aid. GPU-side hot-op attribution still needs Metal capture/counters or a controlled sidecar timing path.
