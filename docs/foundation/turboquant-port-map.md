# TurboQuant Port Map

This document turns the TurboQuant+ donor into a concrete local port map for the clean-room anemll Flash-MoE fork.

## Local evidence first

Current measured operational baseline:
- stock mode
- flash attention on
- KV cache type `q4_0`

Why this is the baseline to beat:
- `q4_0` preserved correct output through 64K context in the local probe matrix
- baseline f16 cache failed at 32K and 64K in this harness
- `q8_0` reached 32K but failed at 64K
- `q4_0` is the strongest built-in compressed-KV baseline currently working on this M4 machine

Reference artifact:
- `results/p6_p7_summary.md`

## Donor patch surface (from TurboQuant+ README)

TurboQuant+ claims changes in these areas:
- `ggml/include/ggml.h`
- `ggml/src/ggml-common.h`
- `ggml/src/ggml-quants.h`
- `ggml/src/ggml-turbo-quant.c` (new file)
- `ggml/src/ggml.c`
- `ggml/src/CMakeLists.txt`
- `ggml/src/ggml-metal/ggml-metal.metal`
- `ggml/src/ggml-metal/ggml-metal-device.m`
- `common/arg.cpp`

## Important naming caveat

The local anemll fork already contains `GGML_TYPE_TQ1_0` and `GGML_TYPE_TQ2_0` in ggml.
These are existing ternary quantization types, not the TurboQuant donor's `TURBO2_0` / `TURBO3_0` / `TURBO4_0` cache types.
Do not confuse or reuse the local `TQ*` names as if they were the donor TurboQuant implementation.

## Local target files

In this repo, the corresponding local targets are:
- `vendor/anemll-flash-llama.cpp/ggml/include/ggml.h`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-common.h`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-quants.h`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml.c`
- `vendor/anemll-flash-llama.cpp/ggml/src/CMakeLists.txt`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m`
- `vendor/anemll-flash-llama.cpp/common/arg.cpp`
- likely also local KV/context init surfaces:
  - `vendor/anemll-flash-llama.cpp/src/llama-context.cpp`
  - `vendor/anemll-flash-llama.cpp/src/llama-kv-cache.h`
  - `vendor/anemll-flash-llama.cpp/src/llama-memory*.{h,cpp}`

## Local constraints already verified

From `src/llama-context.cpp`:
- quantized V cache requires flash attention
- quantized K/V block size must divide the attention head dimensions
- quantized V cache is rejected when flash attention is disabled

Implication:
- TurboQuant must integrate as a flash-attention-compatible KV path
- any type layout must respect head-dimension divisibility and existing context checks

## Port sequence

### Stage 1: plumbing only
Goal:
- make the local fork aware of future TurboQuant cache types without changing runtime behavior yet

Work:
- map donor type additions into local ggml type registration
- map CLI exposure in `common/arg.cpp`
- add build plumbing for a local `ggml-turbo-quant.c` equivalent

Acceptance:
- fork compiles cleanly
- new cache types appear in CLI help
- no behavior changes when the new types are unused

### Stage 2: reference correctness path
Goal:
- add a CPU/reference encode-decode implementation before optimizing Metal

Work:
- norm extraction
- rotation / WHT path
- codebook lookup
- optional QJL/sign path
- unit-style round-trip checks on synthetic tensors and real local KV slices when possible

Acceptance:
- deterministic encode/decode path exists
- no model-load regressions
- correctness checks pass

### Stage 3: Metal path
Goal:
- add the Metal kernels only after the reference path exists

Work:
- encode/decode kernels in `ggml-metal.metal`
- backend validation in `ggml-metal-device.m`
- wiring into local KV cache allocation / usage

Acceptance:
- model runs with the new cache type
- short smoke prompt still passes Lisbon check
- memory footprint improves vs q4_0 or context fit extends beyond q4_0

### Stage 4: performance gate
Goal:
- reject the port if it reproduces the donor's Apple-Silicon slowdown without compensating wins

Acceptance priorities:
1. correctness
2. memory/context fit
3. speed

Reject mainline promotion if:
- throughput is much worse than stock `q4_0`
- quality is visibly worse on short smoke prompts
- context fit does not improve enough to justify slowdown

## Mainline rule

TurboQuant does not replace `q4_0` as the mainline KV path unless it beats the local `q4_0` baseline on this machine in a measurable way.
