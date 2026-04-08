# TurboQuant Experiment 1: Minimum Viable Port

**Date:** 2026-04-06
**Phase:** 2 (TurboQuant KV)
**Plan:** 2026-04-05-stock-first-kv-path.md Task 5

---

## Success Criteria (PRE-COMMITTED)

### Primary: Correctness
- [ ] `--cache-type-k turbo3_0 --cache-type-v turbo3_0` loads without error
- [ ] Quality gate passes: Lisbon + 345 + FizzBuzz
- [ ] No crash on 128-token generation
- [ ] Model produces coherent output (not gibberish)

### Secondary: Memory/Context
- [ ] KV footprint measurable smaller than q4_0 baseline
- [ ] 64K context still fits in 16GB (same as q4_0)

### Tertiary: Speed
- [ ] tok/s measured (no required improvement for Experiment 1)
- [ ] If slower than q4_0 baseline: acceptable for research branch

---

## Rejection Criteria (HARD STOPS)

| Failure | Meaning | Action |
|---------|---------|--------|
| Quality gate fails | Implementation corrupts model | REJECT, debug math |
| OOM at 32K context | Compression insufficient | REJECT for mainline, keep as research |
| Slower by >30% vs q4_0 | Overhead > savings | Move to research branch only |
| Model produces gibberish | WHT/QJL implementation wrong | REJECT, start over |

---

## Minimum Viable Port Scope

**In Scope (Experiment 1):**
1. Port `ggml-turbo-quant.c` (donor → local)
2. Port `ggml-turbo-quant.h` (full public API)
3. Add type handlers in `ggml.c`
4. Build verification

**Out of Scope (Future Experiments):**
- Metal kernels (CPU fallback acceptable for initial test)
- CLI integration (use --cache-type-k flags)
- Perplexity verification (deferred to Experiment 2)
- Long context scaling (deferred to Experiment 2)

---

## Baseline Reference Numbers

From p8_verify_q4_ctx65536:
- Cache: q4_0
- Context: 65536
- Prompt: 50.5 t/s
- Generation: 27.4 t/s
- Status: OK, Lisbon present

Target comparison: turbo3_0 vs q4_0 at same context size.

---

## Port Checklist

- [ ] Copy donor `ggml-turbo-quant.c` → local
- [ ] Copy donor `ggml-turbo-quant.h` → local
- [ ] Add type metadata to `ggml.c` type_info
- [ ] Add quantize hooks to `ggml.c`
- [ ] Verify build compiles
- [ ] Run quality gate
- [ ] Document result vs rejection criteria

---

## Decision Rule After Experiment 1

If all success criteria met:
→ Proceed to Experiment 2 (Metal kernels + perf optimization)

If rejection criteria triggered:
→ Keep code in research branch
→ Mainline stays on stock q4_0/q8_0 KV cache
→ Document findings for future reference
