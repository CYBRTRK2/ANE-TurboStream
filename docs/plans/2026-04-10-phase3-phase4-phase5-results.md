# Phase 3+4+5 Optimization Results

**Date:** 2026-04-10
**Hardware:** M4 MacBook Air 16GB
**Model:** Qwen3.5-35B-A3B-UD-IQ2_M (IQ2_M, 2.7 bpw, ~10.6 GiB)

---

## Defended Baseline: 26 tok/s

| Config | Decode (tok/s) | Prefill (tok/s) | Quality |
|--------|---------------|-----------------|---------|
| q4_0 KV (baseline) | 25.82 avg | 91.8 | PASS |
| turbo3 KV | 25.82 avg | 92.3 | PASS |
| q4_0 KV (N_SG=4 select) | 25.77 avg | — | PASS |
| q4_0 KV (N_SG=4 global) | 24.65 avg | 34.9 | **PREFILL DEGRADED** |

---

## Phase 3: GEMV Kernel Tuning

**Approach:** Increase Metal GEMV threadgroup sizes (N_SG from 2→4) for better M4 GPU occupancy.

**Tested configurations:**

| Config | N_R0 | N_SG | Decode (tok/s) | Impact |
|--------|------|------|---------------|--------|
| Stock (all N_SG=2) | varies | 2 | 25.82 | Baseline |
| Selective N_SG=4 (Q4_K, Q5_K, Q6_K) | 2 | 4 | 25.77 | Neutral (within noise) |
| Selective + N_R0=2 for Q5_K | 2 | 4 | 26.15 | +0.3 (within noise) |
| Global N_SG=4 | varies | 4 | 24.65 | **-4% decode, -62% prefill** |

**Conclusion:** GEMV kernel thread group parameters are NOT the bottleneck. The model is **memory-bandwidth limited** at 120 GB/s. The M4 GPU is already efficiently reading from unified memory; increasing thread group sizes doesn't help because the bottleneck is memory throughput, not GPU occupancy.

---

## Phase 4: ANE Investigation

**Approach:** Test Apple Neural Engine for offloading compute from GPU.

**Benchmarks (CoreML on M4 Air):**

| Operation | Shape | ANE (ms) | GPU (ms) | CPU (ms) |
|-----------|-------|----------|----------|----------|
| Linear | [1000, 2048] | 0.190 | 0.098 | 0.082 |
| Linear | [10000, 2048] | 0.877 | ~0.98 | ~0.82 |

**Key findings:**
1. **ANE is slower than CPU for GEMV** — designed for batched GEMM (convolutions), not single-vector inference
2. **ANE dispatch overhead** (~0.2ms per op) exceeds compute savings for small ops (RMS norm: 0.01ms)
3. **Shared memory bus** — ANE and GPU share 120 GB/s, so concurrent execution splits bandwidth rather than adding it
4. **FP16 requirement** — ANE can't dequantize IQ2_M; would need 4x more memory for FP16 weights
5. **Private API access** — `_ANEInMemoryModel` + `IOSurface` classes resolve on M4, but compilation needs CoreML-generated MIL

**Conclusion:** ANE cannot improve decode performance for this model. It's optimized for batch inference (prefill), not single-token GEMV.

---

## Phase 5: GPU Pipeline Overlap

**Approach:** Measure and potentially overlap CPU graph encoding with GPU compute.

**Timing breakdown per decode token:**

| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| Graph encode (CPU) | 4-12ms (avg ~8ms) | 19% |
| GPU compute (sync_wait) | ~34ms | 81% |
| **Total** | ~42ms | 100% |

**Potential gain from overlap:** If graph encoding overlapped with GPU compute, per-token time would drop from 42ms → 34ms, giving ~29 tok/s.

**Blockers:**
- The CPU needs the previous token's output before it can construct the next graph (autoregressive dependency)
- Command buffer reuse (same graph, different data) is not implemented in llama.cpp
- Would require architectural changes to the `ggml_metal_graph_compute` → `ggml_metal_synchronize` cycle
- Not feasible without significant refactoring of ggml's graph compute loop

**Conclusion:** Pipeline overlap would theoretically give ~3 tok/s, but requires core llama.cpp architectural changes that are beyond the scope of the current optimization pass.

---

## Memory Bandwidth Analysis

The definitive constraint:

- **Model weights:** ~10.6 GiB (IQ2_M @ 2.7 bpw)
- **Data read per token:** ~1.5 GB (dense layers + 8 active experts + LM-head)
- **M4 Air bandwidth:** 120 GB/s
- **Theoretical minimum:** 1.5/120 = 12.5ms → 80 tok/s
- **Observed:** 34ms GPU compute → 3x overhead
- **Practical GEMV overhead factor:** ~3x (typical for single-vector matrix multiplication)
- **Practical ceiling:** ~30-32 tok/s (with perfect pipelining)

---

## Final Ceiling: ~26 tok/s

The **26 tok/s** performance is the practical ceiling for Qwen3.5-35B-A3B (IQ2_M) on M4 Air 16GB because:

1. **All operations are memory-bandwidth bound** — moving any tensor group to CPU gives equal or faster total time
2. **GEMV threadgroup tuning has no effect** — already saturating memory bus
3. **ANE is counterproductive for decode** — slower than CPU for GEMV
4. **Pipeline overlap requires architectural changes** — would give +3 tok/s but not in scope

**Remaining paths to >26 tok/s:**
- IQ1_S quantization (~1.5 bpw) → halve memory read → theoretical ~50 tok/s
- Fewer active experts (e.g., 4 instead of 8) → reduce MoE read by ~50%
- Smaller hidden dim model → proportionally faster
- Command buffer reuse in llama.cpp → +3 tok/s (requires upstream changes)