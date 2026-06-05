# Phase 9: Experiment 1 Complete — TurboQuant Scaffolding with Rejection

**Date:** 2026-04-06
**Experiment:** TurboQuant Minimum Viable Port (Task 5 execution)

---

## Status: PARTIAL / REJECTED FOR MAINLINE

### What Was Accomplished

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| Type enum | TURBO2/3/4_0 present | Same | ✅ Already existed |
| Structs | Mismatched | Donor-aligned | ✅ Updated |
| Quantize functions | 5-line stub | Full 995-line impl | ✅ Ported |
| Header declarations | None | Full public API | ✅ Added |
| `ggml.c` hooks | NULL | Function pointers | ✅ Linked |
| CLI registration | Missing | `turbo3/4/2` accepted | ✅ Working |
| **Metal kernels** | **N/A** | **N/A** | **❌ MISSING** |
| **End-to-end test** | **N/A** | **Backend error** | **❌ BLOCKED** |

### Build Success

- `ggml-base.dylib` compiles (16 warnings, acceptable)
- `llama-cli` accepts `--cache-type-k turbo3` without "unsupported" error
- Full binary rebuilt and functional for standard cache types

### Runtime Rejection

```
ggml-backend.cpp:809: pre-allocated tensor (cache_k_l3 (view)) in a buffer (MTL0) 
that cannot run the operation (SET_ROWS)
```

**Root cause:** Donor implementation includes CPU quantize/dequantize but no Metal kernels for:
- `SET_ROWS` on compressed KV tensors
- Attention path with compressed KV
- Scatter/gather for variable-length sequences

**Decision per pre-committed criteria:** REJECT for mainline. Metal kernels are required for M4 deployment. Research branch checkpoint only.

---

## Regression Check

**Command:** q4_0 @ 65536 context (our p8 baseline)
**Measured:**
- Prompt: 66.3 t/s  
- Generation: 31.5 t/s

**vs p8 baseline (27.4 t/s):** ✓ Within variance, no regression
**Status:** Mainline q4_0/q8_0 remains solid

---

## Pre-Committed Success Criteria vs Reality

| Criterion | Required | Actual | Decision |
|-----------|----------|--------|----------|
| Load without error | Required | CLI accepts turbo3 | ✓ Met |
| Quality gate (Lisbon+345+Fizz) | Required | Blocked before generation | ⚠️ N/A |
| 64K context fit | Required | Blocked before allocation | ⚠️ N/A |
| Measurable speed | Info only | N/A | ⚠️ N/A |
| No hard regression | Required | q4_0 baseline still works | ✓ Met |

**Decision:** Per rejection criteria: "If implementation incomplete (Metal missing), keep as research branch only." Mainline stays on stock.

---

## Files Modified

### Ported from donor → local:
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-turbo-quant.c` (995 lines)
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-turbo-quant.h` (full API)

### Local modifications:
- `ggml-common.h`: Updated `block_turbo4_0` struct (added 3-bit indices + signs)
- `ggml-common.h`: Added `TQ3_1S`, `TQ4_1S` structs for compatibility
- `ggml-quants.h`: Added TurboQuant function declarations
- `ggml.c`: Linked `to_float`/`from_float_ref` handlers
- `arg.cpp`: Added `GGML_TYPE_TURBO*` to `kv_cache_types` array

---

## Next Steps

### Immediate (Mainline)
1. ✓ Verify q4_0 baseline still works (done: 31.5 t/s)
2. Document this experiment in `results/p9_turboquant_scaffolding.md`
3. Consider if Phase 2 is complete or needs different approach

### Future (Research Branch)
1. Extract donor Metal kernels from `vendor/llama-cpp-turboquant/ggml/src/ggml-metal/`
2. Assess completeness: turbo-matrices.h, turbo-wht.h
3. Port Metal if kernels are complete and tested
4. Re-run Experiment 1 with full stack

### Recommendation
Per Plan 2026-04-05-stock-first-kv-path.md:
> "Mainline stays on stock mode until a KV-compression patch proves itself on this machine with local evidence."

**TurboQuant has NOT proven itself.** Research branch only. Continue Phase 2 stock-first work with q8_0/q4_0 context scaling, or proceed to Phase 3 (GPU optimizations).

---

## Artifacts

- Raw output: `results/p9_regression_q4_ctx65536.raw.txt`
- Summary: This file
- Scripts: `scripts/stock_kv_probe.sh`, `scripts/test_turbo3_quality.sh`
