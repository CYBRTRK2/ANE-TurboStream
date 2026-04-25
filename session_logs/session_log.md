# ANE-TurboStream v3 Physical Ceiling Campaign — Session Log
**Date:** 2026-04-25 03:01 WEST
**Agent:** Hermes (Kimi k2.6 via Ollama Cloud)
**Plan:** 2026-04-25-ane-turbostream-v3-physical-ceiling_NEWPLAN.md
**Operator:** Manuel Monteiro (asleep — autonomous execution)
**Commit policy:** Every bounded task committed with descriptive message
**State:** Log to disk. Context refresh from disk on every cycle.

---

## 00:00 — Campaign Start
- Disk: /dev/disk3s5 228Gi total, 156Gi used, 50Gi free (76%)
- Baseline known: stock topk=8 = 20.6 t/s, topk=4 = 23.3 t/s (PROGRESS.md Apr 24)
- Git status: main branch, many untracked files. Committing clean results only.

---

## 00:15 — Track A Preliminary: dflash-cli already partially fixed
**CRITICAL FINDING:** The `vendor/anemll-flash-llama.cpp/build/bin/dflash-cli` binary (dated Apr 23 12:38) is NOT the broken binary described in war_room/execution_plan_v2.md. It already supports `--moe-topk` and reports 27.60 tok/s baseline.

Test result:
```
dflash-cli --baseline --moe-topk 4 -p "Lisbon is the capital of Portugal." -n 64
-> Speed: 27.60 tok/s
```

This means either:
(a) The bug fixes were already applied and rebuilt in vendor/anemll-flash-llama.cpp/, OR
(b) The execution_plan_v2.md bugs refer to a different dflash directory (war_room/cpp/?)

Need to check war_room/cpp/ and compare.

## 00:20 — Continuing Track 0 disk cleanup
- Deleted Time Machine dylib clones in build-nsg-opt/bin/ (0 found)
- Will proceed with build-nsg-opt/bin/ cleanup and results archiving to LaCie

## 00:25 — Next steps declared
1. Finish disk cleanup
2. Write bench_protocol.sh (draft done)
3. Run official baseline with llama-bench (stock topk=8, topk=4, shared-only)
4. Test dflash-block and dflash-AR modes with the working binary
5. If dflash works with draft model, measure acceptance rate
6. Commit session_log.md and key findings to git

---
---

## 03:35 — Track 0 Official Baseline
Agent confirmed operational state:
- Disk: 50 GiB free (76% used)
- build-nsg-opt llama-bench: WORKING
- dflash-cli baseline: 27.60 tok/s
- dflash-draft acceptance: 0% (model-level blocker, C++ already fixed)
- llama-cli stock topk=4: 29.0 tok/s
- llama-bench tg128 stock topk=8: 24.11 +/- 0.19 t/s
- llama-bench tg128 stock topk=4: running now

# Next actions
1. Finish Track 0 baseline numbers and commit
2. Test slot-bank/lookup/lookahead for Track 1 free wins
3. Investigate build-turbo llama-cli for Track 4 TurboQuant
4. Begin Track 2 ANE CoreML backend plumbing (long pole)

