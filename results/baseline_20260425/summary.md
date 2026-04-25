# ANE-TurboStream v3 — Official Baseline — 2026-04-25
**Benchmark Harness:** llama-bench (build-nsg-opt, build da3b409e / 8448)
**Model:** Qwen3.5-35B-A3B-UD-IQ2_M.gguf (10.60 GiB, 34.66 B params)
**Hardware:** Apple M4 MacBook Air, 16 GB unified memory
**Flags:** ngl=99, threads=4, seed=42, ubatch=default, batch=2048
**Campaign Plan:** 2026-04-25-ane-turbostream-v3-physical-ceiling_NEWPLAN.md

| Config | pp512 t/s | tg128 t/s | vs stock topk=8 |
|--------|-----------|-----------|-----------------|
| stock topk=8 | 283.18 +- 1.37 | 23.77 +- 0.42 | baseline |
| stock topk=4 | 340.81 +- 1.73 | 26.02 +- 0.25 | +9.5% |
| shared-only | 539.70 +- 0.23 | 32.89 +- 0.53 | +38.4% |

**Quality gate status (stock topk=4):**
- Lisbon: PASS (verified interactively)
- 23x17: PASS
- Prime 97: PASS

**Key observations:**
- stock topk=4 = 26.02 +- 0.25 t/s. This is the NEW official baseline.
- topk=4 gives +9.5% over stock topk=8 with NO quality regression.
- shared-only = 32.89 t/s (+38.4%) but skips routed experts; quality degrades on hard reasoning.
- The Apr 21 claim of 34.7 t/s is NOT reproducible. Withdrawn.

**Next targets:**
- Track 2 ANE shared-expert dispatch: target >=28 t/s with ANE >0%
- Track 3 dFlash revival: blocked at model level (0% acceptance), needs projector recalibration
- Track 1 free wins: slot-bank mode needs sidecar; llama-lookup/lookahead not yet working
- Track 11A MPSGraphExecutable: clean PR path, estimated +10-170%
