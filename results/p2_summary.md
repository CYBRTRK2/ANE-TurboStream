# Phase 2 Follow-up Summary

Date: 2026-04-04

This phase tightened the slot-bank/oracle picture using a shorter trace and reduced-memory diagnostics.

## Short trace capture

Artifacts:
- `results/p2_short_trace.jsonl`
- `results/p2_short_trace_run.txt`

Config used:
- `--moe-mode slot-bank`
- `--moe-slot-bank 8`
- `-ngl 50`
- `-ub 1 -b 1`
- `-c 128 --ctx-size 128`
- prompt: `The`
- `-n 1 -st`

Observed:
- short trace line count: `548`
- weighted expert coverage at bank 8 is still poor enough to force heavy miss traffic

## Reduced-memory slot-bank comparison

### slot-bank 32
Artifact:
- `results/p2_slotbank32_short.txt`

Observed:
- prompt throughput: `3.3 t/s`
- hit rate: `59.4%`
- streamed bytes: `1.56 GiB`
- residency cold: `1263`
- evictions: `517`
- free GPU memory after run: about `9581 MiB`

### slot-bank 64
Artifact:
- `results/p2_slotbank64_short.txt`

Observed:
- prompt throughput: `3.4 t/s`
- hit rate: `60.2%`
- streamed bytes: `1.53 GiB`
- residency cold: `1743`
- evictions: `3`
- free GPU memory after run: about `8431 MiB`

Interpretation:
- increasing from 32 -> 64 almost eliminates evictions on this short workload
- but short-run prompt throughput did not improve much in this one-token diagnostic
- the bank is large enough to preserve residency, but the measured wall-clock still includes substantial setup/prime cost

## Oracle-all-hit replay on the short trace

Artifacts:
- `results/p2_oracle32_short.txt`
- `results/p2_oracle64_short.txt`
- `results/p2_oracle68_short.txt`
- `results/p2_short_cache_estimator.json`

Observed behavior:
- oracle replay has an off-by-one style slot requirement in this fork for the captured trace:
  - bank 32 -> needs 33
  - bank 64 -> needs 65
  - bank 68 is the first tested value that succeeded

### Successful oracle-all-hit run

At `--moe-slot-bank 68`:
- prompt throughput: `20.7 t/s`
- routed source: `oracle-all-hit`
- hit rate: `100%`
- miss/call: `0.00`
- streamed bytes during replay: `0.00 GiB`
- prime installs before replay:
  - `1746` installs
  - `1.53 GiB`
  - total prime time `1701.183 ms`

Interpretation:
- the short-trace all-hit ceiling is dramatically above the live slot-bank 32/64 reduced-memory runs
- the remaining gap confirms miss/install behavior is still a first-order bottleneck on the live slot-bank path

## Estimator outputs for the short trace

From `results/p2_short_cache_estimator.json`:

Weighted static coverage:
- bank 8: `45.3%`
- bank 32: `88.98%`
- bank 64: `99.93%`
- bank 68: `100%`

Uniform-static misses/token:
- bank 8: `171.3`
- bank 32: `34.5`
- bank 64: `0.214`
- bank 68: `0.0`

Uniform-static resident cost:
- bank 8: `0.281 GiB`
- bank 32: `1.123 GiB`
- bank 64: `2.246 GiB`
- bank 68: `2.386 GiB`

## Current conclusion

The clean-room data now says:
- stock mode is strong (`22.65 t/s` tg32 baseline)
- reduced-memory slot-bank is functional but currently much slower than stock on the tested settings
- for the short trace, useful all-hit behavior begins around an effective bank size of `64+`
- the main next problem is not whether slot-bank works at all, but whether the memory budget and bank geometry can be tuned so the live path approaches the oracle ceiling without collapsing throughput
