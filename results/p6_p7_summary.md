# Phase 6-7 Stock KV Matrix Summary

| cache | ctx | status | prompt_tps | gen_tps | lisbon | failed/error | gpu_context_mib |
|---|---:|---|---:|---:|---|---|---:|
| baseline | 2048 | ok | 54.6 | 24.8 | True | False | 102 |
| q8_0 | 2048 | ok | 52.6 | 22.7 | True | False | 84 |
| q4_0 | 2048 | ok | 64.4 | 27.5 | True | False | 74 |
| baseline | 8192 | ok | 63.7 | 27.5 | True | False | 222 |
| q8_0 | 8192 | ok | 42.7 | 24.4 | True | False | 147 |
| q4_0 | 8192 | ok | 65.8 | 27.3 | True | False | 107 |
| baseline | 16384 | ok | 56.8 | 25.8 | True | False | 382 |
| q8_0 | 16384 | ok | 65.2 | 27.3 | True | False | 232 |
| q4_0 | 16384 | ok | 65.0 | 27.4 | True | False | 152 |
| baseline | 32768 | ok | 0.0 | 0.0 | False | True | 702 |
| q8_0 | 32768 | ok | 27.8 | 3.4 | True | False | 402 |
| q4_0 | 32768 | ok | 47.0 | 27.4 | True | False | 242 |
| baseline | 65536 | ok | 0.0 | 0.0 | False | True | 1342 |
| q8_0 | 65536 | ok | 0.0 | 0.0 | False | True | 742 |
| q4_0 | 65536 | ok | 24.6 | 26.4 | True | False | 422 |

## Conclusions

- `q4_0` is the strongest built-in compressed-KV path tested so far.
- `q4_0` preserved correct Lisbon output through 64K context and kept generation around the mid-20 tok/s range in the short probe.
- baseline f16 cache failed with insufficient GPU memory at 32K and 64K in this harness.
- `q8_0` worked at 32K but failed at 64K.
- `q4_0` reduced GPU context memory materially relative to baseline and `q8_0` at the same context size.
- Practical mainline baseline for future TurboQuant work should now be stock + flash-attn + `q4_0` KV cache.
