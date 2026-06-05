# Experiment 1: TurboQuant Minimum Viable Port - Findings

**Date:** 2026-04-06
**Status:** Partial Success (Registration ✓, Runtime ✗)

---

## What Worked

### ✅ Type Registration
- `GGML_TYPE_TURBO3_0`, `TURBO4_0`, `TURBO2_0` already in `ggml.h` enum
- Function declarations added to `ggml-quants.h`
- Type handlers linked in `ggml.c` type_info table
- CLI cache type registration succeeded in `arg.cpp`

### ✅ Build Integration
- `ggml-turbo-quant.c` (995 lines) compiled with 16 warnings
- `ggml-base.dylib` linked successfully
- `llama-cli` rebuilt and reports cache types:
  ```
  allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1,
  turbo3, turbo4, turbo2
  ```

---

## What Failed

### ❌ Runtime Backend Support
Error at model load:
```
ggml-backend.cpp:809: pre-allocated tensor (cache_k_l3 (view)) in a buffer (MTL0) that cannot run the operation (SET_ROWS)
```

**Root Cause:** Missing Metal kernel implementation for TurboQuant operations.

The CPU quantization code is present, but:
1. No Metal kernels for `SET_ROWS` on compressed tensors
2. No Metal kernels for attention with compressed KV
3. Backend falls back to CPU but buffer is in MTL0 (Metal) memory

---

## Interpretation

| Component | Status | Notes |
|-----------|--------|-------|
| Type enum | ✅ | Already present in anemll fork |
| Struct definitions | ✅ | `ggml-common.h` updated |
| Quantize/dequantize | ✅ | CPU implementation from donor |
| Type_info hooks | ✅ | `ggml.c` linked |
| CLI registration | ✅ | `arg.cpp` accepts `turbo3` |
| Metal kernels | ❌ | Missing - needs port from donor |
| End-to-end test | ❌ | Blocked by Metal gap |

---

## Decision Required

**Option A:** Port Metal kernels (Experiment 2)
- High effort (~500-1000 lines of MSL)
- Unknown if donor Metal kernels are complete
- Risk: may not match anemll fork Metal patterns

**Option B:** Mark as research branch checkpoint
- Current state: "type accepted, backend missing"
- Continue with stock q4_0/q8_0 for mainline
- Return to TurboQuant when Metal kernels ready

**Apple Silicon Team Recommendation:** 
Option B. The quantize/dequantize math is ported and builds. Metal kernels are a separate workstream requiring MSL expertise and hardware testing. Mainline stays on proven q4_0 while Metal work proceeds in parallel.

---

## Next Steps

1. **Immediate:** Document current state in `p9_turboquant_scaffolding.md`
2. **Mainline:** Verify q4_0 baseline still works (regression check)
3. **Research branch:** Extract donor Metal headers, assess kernel completeness
4. **Future experiment:** Port Metal kernels if donor implementation is complete
