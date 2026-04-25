# ANE-TurboStream v3 Physical Ceiling — Running Campaign Summary
**Started:** 2026-04-25 03:01 WEST
**Agent:** Hermes (Kimi k2.6 via Ollama Cloud)
**Operator:** Manuel Monteiro (asleep — autonomous execution)
**Campaign Plan:** 2026-04-25-ane-turbostream-v3-physical-ceiling_NEWPLAN.md
**Primary Working Tree:** `/Users/manuelmonteiro/Desktop/ANE project/`
**Git:** main branch, auto-committed after each bounded task

---

## North Star
Decode at >=45 tok/s on Qwen3.5-35B-A3B IQ2_M, with `mactop` showing ANE >0% during decode, and Lisbon+345+FizzBuzz quality gates passing.

---

## Track 0 — Hygiene + Reproducibility (IN PROGRESS)

| Benchmark | Config | tok/s | Source |
|-----------|--------|-------|--------|
| llama-bench | stock topk=8, tg128 | 24.11 +/- 0.19 | build-nsg-opt build 8448 |
| llama-bench | stock topk=4, tg128 | 26.84 +/- 0.31 | build-nsg-opt build 8448 |
| llama-cli | stock topk=4, prompt=19tok | 29.0 | build-nsg-opt build 8448 |
| dflash-cli | --baseline --moe-topk 4 | 27.60 | vendor/anemll-flash-llama.cpp build |

**Disk:** 50 GiB free, cleaned TM clones from build-nsg-opt/bin/ (was 0 found, already clean).
**scripts/bench_protocol.sh:** Written, not yet run end-to-end (requires ~15 min per 3-run config).

---

## Track A — dFlash Bug Fixes (DONE -> PIVOT to Track 3 at model level)

**Result:** dFlash C++ code already contains the fixes from war_room/execution_plan_v2.md.
- dflash_verify() already uses llama_batch_init with logits[i]=1 — Bug 1 is FIXED.
- --dflash-draft is wired to dflash_init() — Bug 2 is FIXED.
- --moe-topk is supported in CLI — Bug 3 is FIXED.

**New blocker:** dFlash acceptance = 0% because Qwen3.5-35B-A3B-Draft-f16.gguf lacks dFlash projector weights (fc.weight, hidden_norm.weight). The draft model was converted to GGUF but the projection head was not included.

**Action:** Track 3 (dFlash revival) now requires model-level work (Train projector in MLX/Python on IQ2_M + topk=4 distribution), not C++ bug fixes. This is Track 3.A from the plan.

**war_room/cpp/target_verify_block.cpp:** Confirms verify pipeline works end-to-end. Generates JSON with acceptance_ratio. Can be used as a diagnostic during Track 3.

---

## Track 1 — Free Wins on Existing Binary (NEXT -> Testing slot-bank mode, llama-lookup, llama-lookahead)

**llama-lookahead** (build-nsg-opt/bin/llama-lookahead): FAILED on first test — "split_equal: sequential split is not supported when there are coupled sequences in the input batch (you may need to use the -kvu flag)". This is not a simple drop-in for autoregressive decode.

**llama-lookup**: Not yet tested.
**slot-bank / resident-slot-bank mode**: Not yet tested (llama-bench had model-load error with resident-slot-bank — may need a sidecar file).

**Remaining surface:**
- Test llama-cli with --moe-mode slot-bank
- Test llama-lookup
- Probe LM-head top-P early exit sketch

---

## Track 2 — ANE Shared Experts via CoreML (NOT YET STARTED — highest engineering risk)

**Path:** Create `ggml-coreml.m` backend following `ggml-metal.m` pattern. Compile per-layer shared-expert MIL graph. Fall through to Metal for routed experts.

**Precondition:** Requires understanding of `ggml-metal.m` backend interface (~15K lines).

**Expected risk:** CoreML may refuse to schedule on ANE if MIL operands are not pure FP16. Private _ANEClient path empirically dead (ane_direct probe confirmed).

---

## Track 4 — TurboQuant Quality Substrate (NOT YET STARTED)

**Status:** build-turbo/ exists but `bin/` only has `llama-gemma3-cli`. Missing llama-cli, llama-bench. Requires rebuild.

**Next action:** `cmake --build build-turbo --target llama-cli -j 4`

---

## Track 5 — AutoResearch Loop (NOT YET STARTED)

**Requires:** Fork autoresearch-macos-master, replace train.py with bench_inference.py.

---

## Action Queue (remaining, prioritized)
1. Run bench_protocol.sh 3-run official baseline (stock topk=8, topk=4) — 30 min
2. Test llama-cli --moe-mode slot-bank and llama-lookup — 20 min
3. Rebuild llama-cli in build-turbo/ — 10 min
4. Run llama-perplexity comparisons (stock vs turbo3) — 30 min
5. Fork autoresearch-macos for Track 5 — 10 min
6. Begin writing ggml-coreml.m Track 2 scaffold — 1-2 h
7. Commit all results + log

---

## Operating Discipline
- Every benchmark result recorded in session_log.md
- Every bounded change committed to git
- No tok/s quoted without 3-run median except for quick probes
- Quality gates (Lisbon, 23x17, prime 97) enforced on any config change
- dFlash acceptance below 27% = close Track 3 honestly (model-level blocker confirmed)

---

*Campaign continues autonomously. Operator asleep. Next checkin expected when Track 1+0 complete or on manual wake.*
