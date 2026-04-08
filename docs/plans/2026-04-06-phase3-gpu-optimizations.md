# Phase 3: GPU Optimizations Plan

**Date:** 2026-04-06
**Phase:** 3 of Master Plan
**Target:** 22-32 tok/s (from current ~27 t/s baseline)
**Status:** Starting

---

## Master Plan Phase 3 Requirements

From `ANE TurboStream Master Plan.docx`:

> **Goal:** Attack the remaining per-token GPU wait time. After Phase 2, the dominant bottlenecks will be cmd1_wait (GPU wait after attention command buffer), cmd2_wait (GPU wait after MoE command buffer), and lm_head (full vocabulary projection at each decode step).

---

## Task 3.1: Instrument the Timing

**Objective:** Add per-phase timing instrumentation to the Metal backend

**Files to modify:**
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-metal/ggml-metal.m`

**Implementation:**
Mirror what was done in old infer.m:
```objc
NSDate* t0 = [NSDate date];
[cmd_buf_1 waitUntilCompleted];
double cmd1_wait_ms = -[t0 timeIntervalSinceNow] * 1000.0;
```

**Measurements needed:**
1. cmd1_wait: Attention command buffer wait time
2. cmd2_wait: MoE command buffer wait time  
3. lm_head_time: Vocab projection duration
4. Total per-token latency breakdown

**Success Criteria:**
- [ ] Can run `llama-cli` with `--timing` flag
- [ ] Outputs CSV/json with per-phase timings
- [ ] Quality gate still passes (Lisbon+345+Fizz)

---

## Task 3.2: cmd1_wait Reduction

**Objective:** Reduce CPU blocking on attention completion

**Analysis required:**
1. Where is gate projection matmul encoded?
2. Where does CPU read back gate logits for top-K?
3. Is readback inside cmd1 or cmd2?
4. Is there actual overlap potential?

**Constraint:** Do NOT move waitUntilCompleted without tracing full data dependency chain.

**Rejection Criteria:**
- If moving wait causes quality gate to fail → reject change
- If no measurable cmd1_wait reduction → document ceiling

---

## Task 3.3: LM-Head Optimization

**Objective:** Reduce vocabulary projection bottleneck

**Approaches:**
1. **Chunked parallel dispatch:** Split 152K vocab across multiple encoders
2. **Top-P early exit:** Stop computing logits once cumulative mass > top-P

**Pre-check:** Read `results/` for previously rejected LM-head approaches. Do not re-propose `--gguf-lm-head` or `--nax`.

---

## Pre-Committed Success Criteria (Experiment 3A: Instrumentation)

| Criterion | Required | Measurement |
|-----------|----------|-------------|
| Instrumentation compiles | Required | Build succeeds with timing hooks |
| Quality gate passes | Required | Lisbon+345+Fizz after timing hooks |
| Timing data validates | Required | cmd1/cmd2/lm_head times measurable |
| No regression | Required | tok/s within 5% of baseline |

---

## Baseline Reference

From Phase 2:
- Cache: q4_0  
- Context: 65536
- Generation: 26.4 t/s (p9 regression check)
- Quality: PASS (Lisbon confirmed)

---

## Decision Rule

Per Master Plan:
> "Any change that causes quality gate to fail: immediate rejection."
> "Any change where the targeted phase does not improve: reject the specific change."
> "If neither cmd1 nor lm_head can be improved: document the ceiling and move to Phase 4."

**Phase 3 is important but not a blocker for Phase 4 (ANE).**

