# Phase 8 Stage-1 TurboQuant Scaffolding Summary

Date: 2026-04-05

## Goal

Begin Stage 1 of the TurboQuant path with no runtime behavior change when unused.

## Donor acquisition

Cloned donor fork for direct source inspection:
- `vendor/llama-cpp-turboquant`
- branch: `feature/turboquant-kv-cache`
- commit: `bc05a6803`

This donor confirmed that the real TurboQuant port is substantially larger than a simple cache-type alias. It includes:
- new cache types (`TURBO2_0`, `TURBO3_0`, `TURBO4_0`)
- new `ggml-turbo-quant.c`
- graph changes in attention paths
- KV cache allocation changes
- flash-attn auto-enable / shape logic
- Metal kernels and backend validation work

## Local scaffolding added

Files added to the active fork:
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-turbo-quant.h`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-turbo-quant.c`

Build plumbing updated:
- `vendor/anemll-flash-llama.cpp/ggml/src/CMakeLists.txt`

Current behavior:
- only a stub symbol exists: `ggml_turbo_quant_stage1_stub_present()`
- no CLI changes
- no new cache types exposed
- no runtime behavior changes when unused

## Naming caveat documented

Important local note added to:
- `docs/foundation/turboquant-port-map.md`

Reason:
- local ggml already contains `TQ1_0` / `TQ2_0`
- these are not the donor TurboQuant cache types
- they must not be confused with `TURBO2_0` / `TURBO3_0` / `TURBO4_0`

## Build verification

Reconfigured and rebuilt successfully:
- target `ggml-base`
- target `llama-cli`

## Regression check

Post-build q4_0 mainline verification:
- artifact: `results/p8_verify_q4_ctx65536.summary.md`
- result:
  - status `ok`
  - prompt `50.5 t/s`
  - generation `27.4 t/s`
  - Lisbon present
  - 64K context still works with q4_0

## Conclusion

Stage 1 scaffolding is in place.
No new TurboQuant functionality exists yet, but the active fork now has:
- donor repo locally available for direct comparison
- a reserved local source file/build slot for future TurboQuant implementation
- a verified no-regression build after adding the scaffolding

Next logical step:
- inspect and port the minimum type/CLI/build plumbing for true turbo cache types, while keeping runtime disabled until the first reference-correctness path is ready.
