# Phase 5 Stock-First KV Compression Audit

Date: 2026-04-05

Goal: establish the next mainline path after the slot-bank pivot by testing built-in KV cache quantization and auditing the TurboQuant+ donor surface.

## Local runtime constraints found

Relevant local source lines:
- `src/llama-context.cpp:2742-2745`
  - quantized V cache requires Flash Attention
- `src/llama-context.cpp:5407-5430`
  - quantized K/V cache also requires head dimensions that divide the quant block size, and quantized V is rejected if flash attention is disabled

Implication:
- any stock-path KV compression work must respect flash-attn and block-size constraints in the current fork

## Built-in cache-type smoke tests

### q8_0 KV cache
Artifact:
- `results/p5_stock_q8_smoke.txt`

Command shape:
- stock mode
- `-ngl 99`
- `-ctk q8_0 -ctv q8_0`
- `--reasoning off -st`

Observed:
- output still contains `Lis...` for Lisbon
- prompt throughput: `20.7 t/s`
- generation throughput: `14.8 t/s`

### q4_0 KV cache
Artifact:
- `results/p5_stock_q4_smoke.txt`

Observed:
- output still contains `Lis...` for Lisbon
- prompt throughput: `25.6 t/s`
- generation throughput: `16.3 t/s`

## llama-bench issue

Attempts to benchmark q8_0 / q4_0 cache types with `llama-bench` failed at context creation, even when flash attention was explicitly enabled.

Artifacts / failed outputs:
- `results/p5_stock_q8_cache.md`
- `results/p5_stock_q4_cache.md`
- `results/p5_stock_q8_cache_fa.md`
- `results/p5_stock_q4_cache_fa.md`

Result:
- `main: error: failed to create context with model ...`

Interpretation:
- in this fork, `llama-cli` can run short stock quantized-cache tests, but `llama-bench` is not currently a reliable harness for those cache-type experiments

## TurboQuant+ donor audit

From `papers:references:repos/turboquant_plus-main/README.md`:
- TurboQuant+ modifies these llama.cpp surfaces:
  - `ggml/include/ggml.h`
  - `ggml/src/ggml-common.h`
  - `ggml/src/ggml-quants.h`
  - `ggml/src/ggml-turbo-quant.c` (new)
  - `ggml/src/ggml.c`
  - `ggml/src/CMakeLists.txt`
  - `ggml/src/ggml-metal/ggml-metal.metal`
  - `ggml/src/ggml-metal/ggml-metal-device.m`
  - `common/arg.cpp`
- donor claims quality is good (`+1.4%` PPL vs q8_0 for turbo3)
- donor also explicitly reports a major speed regression on Apple Silicon due to inverse rotation overhead in the dequant path

Interpretation:
- TurboQuant+ is useful as an algorithm and integration donor
- but not as a drop-in speed win
- if we port it, we should expect a quality/memory success path first, not a throughput success path first

## Mainline recommendation

The stock-first path should now proceed in this order:
1. use stock llama.cpp as the serving baseline
2. treat built-in q4_0/q8_0 cache types as the immediate operational baseline for compressed-KV experiments
3. audit TurboQuant+ for portable algorithm pieces and identify the minimum patch surface in the local anemll fork
4. design the first bounded stock-path KV experiment around correctness / memory / context-fit first, and speed second
