# Phase 4 Cold/Warm/Oracle Decision Summary

Date: 2026-04-04

Goal: determine whether live slot-bank performance improves meaningfully on repeated runs, or whether the project should pivot back to stock-first work.

## Repeated live slot-bank 68 runs

Artifacts:
- `results/p4_slotbank68_run1.txt`
- `results/p4_slotbank68_run2.txt`

Config:
- `--moe-mode slot-bank`
- `--moe-slot-bank 68`
- `-ngl 50`
- `-ub 1 -b 1`
- `-c 128 --ctx-size 128`
- prompt `The`
- `-n 1 -st`

### Run 1
- prompt throughput: `3.4 t/s`
- source time: `1801.241 ms`
- install time: `1915.269 ms`
- upload time: `111.262 ms`
- hit rate: `60.2%`
- bytes streamed: `1.53 GiB`
- cold installs: `1746`
- evictions: `0`

### Run 2
- prompt throughput: `3.3 t/s`
- source time: `1850.543 ms`
- install time: `1967.947 ms`
- upload time: `114.749 ms`
- hit rate: `60.2%`
- bytes streamed: `1.53 GiB`
- cold installs: `1746`
- evictions: `0`

## Temporal prefetch check

Artifact:
- `results/p3_slotbank68_prefetch_short.txt`

Observed:
- prompt throughput: `3.3 t/s`
- prefetch temporal created no meaningful throughput improvement

Interpretation:
- prefetch does not address the dominant cost on this workload

## Repeated oracle-all-hit control

Artifacts:
- `results/p4_oracle68_run1.txt`
- `results/p4_oracle68_run2.txt`

Config:
- `--moe-mode oracle-all-hit`
- `--moe-slot-bank 68`
- same short trace and runtime geometry as live tests

### Oracle run 1
- prompt throughput: `21.3 t/s`
- prime installs: `1746`
- prime bytes: `1.53 GiB`
- prime total: `1761.973 ms`
- routed replay bytes after prime: `0.00 GiB`

### Oracle run 2
- prompt throughput: `20.3 t/s`
- prime installs: `1746`
- prime bytes: `1.53 GiB`
- prime total: `1798.004 ms`
- routed replay bytes after prime: `0.00 GiB`

## Decision

The repeated-run evidence says:
- live slot-bank 68 does **not** become meaningfully faster on an immediate repeat run
- source/install overhead remains almost unchanged across repeated runs
- temporal prefetch does not move the needle
- the gap between live slot-bank (`~3.3-3.4 t/s`) and all-hit oracle (`~20.3-21.3 t/s`) remains huge

## Conclusion

For this machine and current implementation state:
- the slot-bank concept is valid in theory
- but the live path is still dominated by miss/install behavior
- repeated runs do not show a practical warm-run rescue

Recommended project direction now:
1. treat stock llama.cpp as the primary serving path
2. treat slot-bank as a research lane, not the mainline speed path
3. move the next mainline optimization focus to stock-compatible work (e.g. KV compression / TurboQuant-style path, or other stock-path improvements)
4. return to slot-bank only if a very specific install/source-path optimization target is identified from donor code and can be tested as a bounded experiment
