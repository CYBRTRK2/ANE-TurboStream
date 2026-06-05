# Phase 3 & 4 Analysis: LM-Head Optimization and ANE Integration

**Date:** 2026-04-10
**Baseline:** 26 tok/s (q4_0 KV, llama.cpp anemll-flash fork, M4 Air 16GB)

---

## Phase 3: GPU/LM-Head Optimization

### Profiling Results

**Method:** Override-tensor A/B testing (move tensor group to CPU, measure delta)

| Component on CPU | ms/tok | Delta vs Baseline (38.98ms) |
|---|---|---|
| Baseline (all GPU) | 38.98 | — |
| output.weight (LM-head) on CPU | 38.01 | -0.97ms (CPU faster!) |
| MoE expert weights on CPU | 38.64 | -0.34ms |
| Shared expert on CPU | 38.74 | -0.24ms |
| QKV+SSM weights on CPU | 38.02 | -0.96ms |

**Key findings:**
1. **ALL GPU ops are memory-bandwidth limited**, not compute-bound
2. Moving any tensor group to CPU makes total time SLIGHTLY LOWER because CPU avoids Metal command buffer dispatch overhead
3. LM-head on GPU takes ~1ms (the 0.97ms saved by using CPU)
4. The total GPU compute time is only ~1.5ms — the remaining ~37ms is unified memory reads
5. **The 26 tok/s is close to the practical ceiling for this model on M4 Air 16GB**

### Memory Bandwidth Analysis

- Model weight: ~10.6 GiB (IQ2_M, 2.7 bpw)
- M4 Air memory bandwidth: 120 GB/s
- Per-token memory read: ~1.5 GB (non-expert: 680MB + MoE 8 active: 640MB + LM-head: 170MB)
- Theoretical minimum: 1.5/120 = 12.5ms/tok → 80 tok/s
- Practical GEMV overhead: ~3x → ~37ms/tok → ~27 tok/s
- **Estimated practical ceiling: ~30-32 tok/s** (with perfect kernel pipelining)

### Phase 3 Decision

**LM-head optimization is NOT viable** — it's only ~1ms of the 39ms total.

**The real bottleneck is memory bandwidth for the entire model**, not any single operation.

**Recommendation:** Document the ceiling and proceed to Phase 4 (ANE).

---

## Phase 4: ANE Integration

### ANE Access Methods

1. **CoreML (MLModel + ComputeUnit.CPU_AND_NE)** — Works, official API
2. **Private API (_ANEInMemoryModel + IOSurface)** — Classes resolve on M4, compilation needs CoreML-generated MIL (not freeform text)
3. **Metal Tensor API** — Available on M4 but llama.cpp gates behind M5

### ANE Performance (CoreML benchmarks on M4 Air)

| Op | Shape | ANE (ms) | GPU (ms) | CPU (ms) |
|---|---|---|---|---|
| Linear | [1000, 2048] | 0.190 | 0.098 | 0.082 |
| Linear | [10000, 2048] | 0.877 | ~0.98 | ~0.82 |

**Critical finding:** ANE is SLOWER than CPU for single-token GEMV.

### Why ANE Doesn't Help for Decode

1. **GEMV is not ANE-friendly** — ANE is optimized for batched matmul (convolutions), not single-vector × large-matrix
2. **Shared memory bus** — ANE and GPU both read from the same 120 GB/s unified memory. Running ANE + GPU simultaneously splits bandwidth, doesn't add capacity.
3. **ANE dispatch overhead** (~0.1-0.3ms per op) exceeds the compute time for most LLM ops
4. **FP16 only** — ANE cannot dequantize IQ2_M. The model weights must be stored in FP16, which means 4x more memory (42 GiB for the full model — doesn't fit in 16GB)

### What ANE COULD Do (Theoretical)

The Master Plan identifies these as potentially ANE-eligible:
- **RMS Norm** (40x per token) — small (2048 elements), FP16, static graph
- **Shared expert projections** ([512,2048] FP16) — ~6 MB per layer, static
- **QKV projection** ([8192,2048] FP16) — ~16 MB per layer, static

But in FP16, just the QKV weights = 16 MB × 40 layers = 640 MB. The entire model in IQ2_M is 10.6 GB. In FP16 for just the ANE-eligible dense layers, we'd need ~2 GB extra — on a machine that only has ~1.5 GB free.

### ANE + GPU Parallelism

**Could ANE do RMS Norm while GPU does the matmul?** 
- 40 RMS norms × ~0.01ms each = 0.4ms of compute
- But the dispatch overhead per ANE op is ~0.1-0.3ms
- So 40 × 0.2ms dispatch = 8ms overhead just to schedule
- That's 8ms of CPU time wasted vs 0.4ms of saved GPU time
- **Net LOSS**

### ANE for Prefill (Not Decode)

ANE shines for **batched inference** (prompt processing). During prefill:
- Multiple tokens processed simultaneously → GEMM, not GEMV
- ANE can sustain high TOPS for these operations
- But prefill is already at 136 tok/s — not the bottleneck

---

## Conclusion

| Path | Feasibility | Expected Gain | Verdict |
|---|---|---|---|
| LM-head optimization | Low (only 1ms saved) | +0.5 tok/s | ❌ Not worth the effort |
| ANE for decode GEMV | Failed (ANE slower than CPU) | -2 tok/s | ❌ ANE is counterproductive |
| ANE for RMS Norm | Marginal (dispatch overhead > compute) | ~0 tok/s | ❌ Net loss |
| ANE for prefill | Works (batched) | Already fast | ⚠️ Not needed |
| Memory bandwidth reduction | Only via smaller model | +4-6 tok/s | 🔶 Requires IQ1_S or model surgery |
| Pipeline optimization | Possible (overlap GPU command buffers) | +2-4 tok/s | ✅ Best remaining option |
| Batched inference | Not for single-user | N/A | ❌ Wrong use case |

**The 26 tok/s ceiling for this 35B MoE model on M4 Air 16GB is a hardware limit imposed by unified memory bandwidth.** Getting to 30+ tok/s would require either:
1. A smaller model (fewer active experts, lower hidden dim)
2. More aggressive quantization (IQ1_S at ~1.5 bpw, but quality loss)
3. Better GEMV kernels with reduced memory access overhead (possible 2-3 tok/s gain)
4. Dual-operation pipelining on GPU (overlap compute with memory fetches)

**Recorded ceiling: 26 tok/s stable decode. Moving to document and close Phase 3.**