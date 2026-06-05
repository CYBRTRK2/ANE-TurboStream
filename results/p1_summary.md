# Phase 1 Summary (work in progress)

Date: 2026-04-04
Model: `/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf`
Sidecar path: `/Users/manuelmonteiro/Desktop/ANE project/results/sidecar/qwen35`

## Fresh sidecar extraction

Command:
`./.venv-tools/bin/python vendor/anemll-flash-llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py extract --model /Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf --out-dir /Users/manuelmonteiro/Desktop/ANE project/results/sidecar/qwen35 --force`

Observed result:
- wrote `120 tensors across 40 layers`
- total exact bytes copied: `9646899200`
- manifest written successfully
- verification completed successfully: `verified 120 Flash-MoE sidecar entries against 1 GGUF file(s) using metadata+bytes`

## Disk-space incident

A first extraction attempt failed because the volume ran out of space. A partial `results/sidecar/` tree was removed and extraction then succeeded.

## Archived sidecar check

An archived sidecar from the old repo was tested as an interim artifact and proved unreliable:
- slot-bank read failures from `layer_026.bin`
- not suitable as the active Phase 1 artifact

Conclusion:
- use the freshly extracted sidecar only

## Slot-bank diagnostics

### slot-bank 32 (fresh sidecar, `-ngl 999`)
Result:
- process was killed by the OS (`Killed: 9`) before useful output was produced

Interpretation:
- this configuration is not yet viable on the current memory budget in the clean workspace

### slot-bank 8 diagnostic (`-ngl 50 -ub 1 -b 1 -c 256 --ctx-size 256`)
Artifact:
- `results/p1_slotbank8_diag.txt`

Observed result:
- prompt throughput: `2.8-3.0 t/s`
- generation throughput: `4.5-5.4 t/s`
- output began generating correctly (`The capital of Portugal ...`)
- runtime summary showed:
  - hit rate: `31.2%`
  - bytes streamed: `3.98 GiB`
  - residency cold: `320`
  - evictions: `4214`

Interpretation:
- slot-bank works in a reduced-memory diagnostic regime
- locality is poor at bank size 8 for this trace
- miss/install cost dominates the run

## Trace capture

Trace artifact:
- `results/p1_slotbank8_trace.jsonl`
- line count: `824`

## Oracle-all-hit diagnostics

Artifacts:
- `results/p1_oracle_all_hit_diag.txt`
- `results/p1_oracle_all_hit_trace_diag.txt`
- `results/p1_oracle_all_hit_trace16_diag.txt`
- `results/p1_oracle_all_hit_trace32_diag.txt`

Observed behavior:
- oracle mode requires a trace file
- with the captured trace, oracle context initialization failed unless the slot-bank could hold the trace's peak per-layer working set
- failures observed:
  - needs `9` slots in layer 0 when configured with `8`
  - needs `17` slots in layer 0 when configured with `16`
  - needs `33` slots in layer 0 when configured with `32`

Interpretation:
- the captured trace exceeded the configured slot-bank size at layer 0 under oracle replay
- next useful step is likely a larger-bank oracle replay (e.g. 64) if memory allows, or a shorter/smaller trace designed for ceiling measurement

## Cache-estimator findings from captured trace

Artifact:
- `results/p1_cache_estimator.json`

Key outputs from the 21-token trace:
- resident cost:
  - bank 8: `0.281 GiB`
  - bank 16: `0.562 GiB`
  - bank 32: `1.123 GiB`
  - bank 64: `2.246 GiB`
- weighted static coverage:
  - bank 8: `36.0%`
  - bank 16: `55.7%`
  - bank 32: `79.0%`
  - bank 64: `98.2%`
- uniform-static misses per token:
  - bank 8: `200.8`
  - bank 16: `138.9`
  - bank 32: `65.8`
  - bank 64: `5.7`

Important implication:
- on this trace, `slot-bank 32` is theoretically much better than `slot-bank 8`, but still far from all-hit behavior
- `slot-bank 64` is the first bank size that approaches an all-hit ceiling on this workload, at roughly `2.25 GiB` routed-bank residency

## Current status

- Phase 0 baseline: complete
- Fresh sidecar extraction: complete and verified
- Slot-bank diagnostics: complete for initial reduced-memory trace path
- Oracle ceiling measurement: runtime oracle replay exposed the required slot-count behavior, and the cache estimator now provides a usable theoretical ceiling map for 8/16/32/64
