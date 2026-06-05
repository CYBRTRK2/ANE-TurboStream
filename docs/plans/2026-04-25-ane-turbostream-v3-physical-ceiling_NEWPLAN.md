# ANE-TurboStream v3 — Physical Ceiling Campaign
**Date:** 2026-04-25
**Hardware:** M4 MacBook Air, 16 GB unified memory
**Model:** Qwen3.5-35B-A3B-UD-IQ2_M.gguf (10.6 GB, in RAM)
**Runtime:** llama.cpp build b8448-da3b409e (anemll Flash-MoE fork) at `build-nsg-opt/bin/`
**Author:** Apple Silicon AI Engineering Team
**Supersedes:** all prior plan docs in `docs/plans/`. Read those for history; do not re-execute them.

---

## 0. Identity — Read this before touching anything

You are not a generic LLM. For the duration of this campaign you are a member of the **Apple Silicon AI Engineering Team**: the people who built the ANE compiler, the Metal Performance Shaders framework, the CoreML pipeline, and ship production on-device inference across every Apple device. You think in **dispatch latency, memory bus contention, kernel launch overhead, SRAM tile boundaries, and IOSurface lifecycles.**

You do not say "this should be faster." You say "this op runs in 0.41 ms on GPU but its weights are 8 MB FP16 in static shape — dispatching to ANE saves 0.27 ms after subtracting two 0.095 ms XPC dispatch costs."

You do not propose changes you have not measured. You do not believe other agents' claims. You believe `mactop`, `llama-bench`, and `gettimeofday()`.

You are also expected to **push back**. If a track is sunk cost, say so on paper. If a number doesn't reproduce, retire it. The user has explicitly granted you the right to disagree.

---

## 1. North Star

> **Decode at ≥45 tok/s on Qwen3.5-35B-A3B IQ2_M, on this M4 Air, with the Lisbon + 345 + FizzBuzz quality gate passing, and `mactop` showing ANE >0% during decode.**

That last clause is non-negotiable. The project is called ANE-TurboStream. Every other LLM engine on Apple Silicon shows ANE = 0%. If we hit 45 t/s with ANE still at 0% we have *not* shipped the thesis — we've just tuned llama.cpp.

**Physical ceiling math (for sanity):**
- M4 unified bandwidth: 120 GB/s
- Active params/token at IQ2_M ≈ 1.5 GB
- BW ceiling: ~80 t/s ideal, ~45–55 t/s realistic with KV + LM-head + dispatch
- Current: 23.3 t/s (topk=4, Apr 24, reproducible)
- Headroom: ~2×. Below the ceiling, "physically impossible" is not the constraint — engineering is. We do not stop until we hit the wall.

---

## 2. Verified reality (Apr 24–25 2026)

**Reproducible:**

| Config | Decode tok/s | Quality |
|---|---|---|
| stock topk=8 | 20.6 | ✅ all gates |
| **topk=4 (daily driver)** | **23.3 (+13%)** | ✅ all gates |
| topk=2 | 25.2 | ❌ 23×17 hallucinates 36.200 |
| topk=1 | 30.1* | 💥 chat parser abort on math |
| shared-only | 37.5 | ⚠️ no routed experts — upper bound only |

**Retired / discredited:**
- The Apr 21 PROGRESS.md claim of 34.7 t/s for topk=4 is **not reproducible** and is hereby withdrawn. Same binary, same model, same flags, different day → 1.5× gap. The Apr 21 number is a thermal artifact or a measurement bug; it shall not be cited.

**Dead-ends to leave dead unless explicitly revived per Track 3:**
- Standard llama-speculative AR + a different-architecture draft → blocked permanently by M-RoPE / GatedDeltaNet position incompat.
- DFlash with default reference proposer + IQ2_M target → 0–1 / 15 acceptance on the Lisbon prompt. Cause analysis below; revival is conditional.

**Operational blockers:**
- **Disk 5.9 GB free / 98 % used.** First action of Track 0 is reclaim. Many duplicate `libggml 2.dylib` / `libllama 3.dylib` Time Machine artifacts in `build-nsg-opt/bin/`.
- 130 entries in `results/`; most from Phase 0/1/2 are no longer load-bearing.
- `war_room/` carries DFlash sprint artifacts that should be archived to LaCie before any new artifacts.

---

## 3. Strategic synthesis — the four levers, fused

The user has asked, correctly, for a plan that does not pick one lever. We squeeze the M4 Air with **four levers running concurrently, each addressing a different bottleneck**, and one **AutoResearch loop** orchestrating them.

| Lever | Bottleneck it attacks | Estimated gain |
|---|---|---|
| **ANE shared-expert dispatch (Phase 4 of master plan)** | GPU dispatch + routed/shared expert serialization | +20–30 % decode |
| **dFlash revival (recalibrated against IQ2_M)** | Sequential decode latency | +30–60 % via accepted-block speedup |
| **TurboQuant — KV cache quality, not speed** | Perplexity headroom for aggressive levers | enables topk=4 + dFlash without quality regression |
| **EvoDistill (ANE-shaped fine-tune)** | Model is hardware-blind | indirect: makes routing predictable → ANE + dFlash both win |
| **AutoResearch loop (Karpathy-style)** | Human bandwidth | meta — finds wins humans miss; closes science loop |

**The synthesis the previous plans missed:**
- ANE dispatch wants **static, predictable routing.**
- dFlash wants **a draft whose distribution matches the target.**
- EvoDistill, when its fitness function rewards low-entropy routing and ANE-friendly shapes, gives both ANE and dFlash what they need.
- AutoResearch closes the loop: hypothesis → metal kernel diff → bench → accept/reject → next hypothesis. No human in the inner loop.

This is the integration the project has been missing. Every prior plan optimized one lever at a time.

---

## 4. Track 0 — Hygiene (TODAY, ~2 h)

**Why first:** disk pressure (5.9 GB free) blocks every other track. Reproducibility is broken. We cannot ship a campaign on top of unreliable numbers.

**Steps:**

1. **Disk reclaim.**
   - Delete duplicate dylibs in `build-nsg-opt/bin/`: every `* 2.dylib`, `* 3.dylib` (Time Machine xattr clones). Verify md5 against the canonical before delete.
   - Move `war_room/artifacts/*.bin` and `war_room/logs/` to `/Volumes/LaCie/ANE_archive/2026-04-25-war-room/`.
   - Move `results/` entries older than Apr 21 to `/Volumes/LaCie/ANE_archive/2026-04-25-results/`. Keep `topk*_quality.txt`, `topk_focused.log`, `topk_sweep_2026042*` only.
   - Target: ≥30 GB free post-cleanup.

2. **Reproducibility protocol (commit to repo as `scripts/bench_protocol.sh`).**
   - 3 runs, each preceded by a 30 s warm-up generation that is discarded.
   - Log `pmset -g thermlog` and `mactop --json` snapshot at each run.
   - Fixed seed (`--seed 42`), `--temp 0`, `-n 128`, fixed prompt set.
   - Output: `results/baseline_YYYYMMDD/{run1,run2,run3}.json` + `summary.md` with median + IQR.
   - **No tok/s number is allowed in PROGRESS.md unless produced by this script.**

3. **Re-run baseline with the new protocol** for `stock topk=8`, `topk=4`, `shared-only`. Publish median and IQR. **This becomes the official baseline.**

4. **Update `PROGRESS.md`:**
   - Strike the Apr 21 numbers with a one-line note: "withdrawn 2026-04-25, non-reproducible."
   - Mark DFlash sprint of Apr 24 as terminated for this generation; revival happens under Track 3 with explicit preconditions.

**Track 0 gate:** ≥30 GB free, 3-run median baseline published, PROGRESS.md cleaned. Do not begin any other track until this gate is closed.

---

## 5. Track 1 — Free wins on the existing binary (1–3 days)

`build-nsg-opt/bin/` already contains compiled binaries we have never run end-to-end. They are speculative-decoding paths that **do not use a different-architecture draft** and therefore are immune to the M-RoPE blocker.

### 1.1 `llama-lookahead` (Yifei Li et al., 2024)
- N-gram parallel decoding. No draft model. No M-RoPE conflict. Already compiled.
- Test on factual + math + code prompts at `--moe-topk 4`.
- Expected: +15–30 % on cache-hit-friendly outputs (factual, code).
- **If this works, it is the new daily driver and we keep going.** No code written.

### 1.2 `llama-lookup` + `llama-lookup-create`
- Prompt-conditional n-gram speculative. Build the lookup cache once, reuse per session.
- Expected: +10–20 % on prompts with high local repetition.

### 1.3 LM-head top-P early exit
- 152 K vocab × 7168 hidden = ~1.1 GB read per token at decode → bandwidth-bound.
- The master plan called for top-P chunked exit; never implemented.
- Sketch: argmax in chunks of 16K, accumulate prob mass, halt at 0.99.
- Expected: +5–10 %, quality-neutral.
- File to touch: `vendor/anemll-flash-llama.cpp/ggml/src/ggml-metal/ggml-metal.m`, the `lm_head_mul_mat` path.

### 1.4 cmd1 / cmd2 overlap probe
- Phase 3 instrumentation already exists (Apr 9–10). Read the data; if `cmd1_wait` > 1 ms and is not data-dependent on `cmd2`, schedule overlap.
- Expected: +5 % only if the dependency analysis is favorable.

**Track 1 gate:** lookahead measured (gain or no gain logged), top-P prototype exists, cmd1 dependency mapped. **Whatever wins on quality gate becomes the new `daily_driver.sh`.**

---

## 6. Track 2 — Phase 4 ANE: shared experts on the Neural Engine (5–10 days)

**This is the thesis.** The original master plan (`docs/foundation/ANE TurboStream Master Plan.docx`) put this at Phase 4. The project skipped to topk tuning. We now do it.

### Why shared experts first
The ANE dispatch rule from the master plan is: FP16 weights + static graph + GPU op time > 0.285 ms. Shared experts are the only sub-graph that satisfies all three:
- Always active (every token, every layer) → static.
- ~1.2 GB FP16 if dequantized at load → fits ANE-eligible.
- Their gate+up+down ops dominate the dense compute on Qwen3.5-35B → > 0.285 ms each.

Routed experts fail rule 2 (dynamic). LM-head fails rule 3 (BW-bound, not compute-bound). QKV is borderline; revisit after shared experts work.

### Path A — `ggml-coreml.m` backend (recommended)
1. **Plumbing:** add `ggml/src/ggml-coreml.m` mirroring `ggml-metal.m`'s backend interface. Register as `GGML_BACKEND_TYPE_COREML`.
2. **MIL graph compile:** for each layer, compile the shared expert sub-graph (`ffn_norm → shared_gate → shared_up → silu → shared_down → residual`) as one MIL `MLProgram`. **Compile per-layer not per-op** — the ANE compiler fuses across the graph; per-op compilation is why prior attempts got 0% ANE utilization.
3. **Dequantize once at load:** shared experts read from GGUF in IQ2_M; convert to FP16 on first load and cache in an `IOSurface`-backed `MLMultiArray`. Memory cost: ~1.2 GB. We have headroom because slot-bank is not in use.
4. **Dispatch:** at `ggml_backend_coreml_graph_compute`, route shared expert nodes to the compiled MLProgram via `MLPredictionFromBatch`. All other ops fall through to Metal.
5. **Validation gates (in this order):**
   - Compile succeeds.
   - First inference produces correct logits (token-by-token compare to GPU-only).
   - `mactop` shows ANE > 0 % during decode. **Anything less is a failure regardless of tok/s.**
   - Quality gate: Lisbon + 345 + FizzBuzz pass.
   - Decode tok/s ≥ Track 1 best.

### Path B — Private `_ANEClient` (only if A is insufficient)
Per the master plan's Phase 4 path B and `papers:references:repos/` notes on `maderix/ane`. Reverse-engineered MIL compile via `_ANECompiler`, dispatch via `_ANEClient` + `IOSurface`. Higher control, App-Store-incompatible. **Do not start here. Path A first; switch only if CoreML refuses to schedule on ANE.**

### Path A reality check
CoreML auto-routing decides based on op shape and the device's `MLComputeUnits`. We must explicitly set `.cpuAndNeuralEngine` (not `.all`) to *force* ANE consideration. Test that the model loaded with `.all` actually picks ANE; if not, the diagnostic is whether MIL operands are pure FP16 with no shape ambiguity.

### Track 2 gate
`mactop` ANE % > 0 sustained during decode for ≥30 s, quality gates pass, decode ≥ 28 t/s. **The screenshot of `mactop` showing ANE > 0 is the artifact that proves the project shipped.**

---

## 7. Track 3 — dFlash revival, *correctly* (parallel to Track 2; 7–14 days)

The Apr 24 sprint log is honest: 0/15 and 1/15 acceptance on a single Lisbon prompt. I called this dead in my audit. **The user has correctly pushed back. Revive it under specific preconditions.**

### Why it failed (root cause, not symptom)
1. **Quantization mismatch.** DFlash was trained against an FP16 / BF16 target distribution. Our target is IQ2_M. The proposer's projector head was calibrated against the pre-quantization logit distribution. IQ2_M perturbs logits enough that the proposed block is mis-aligned.
2. **topk override mismatch.** DFlash assumed default topk=8 routing. We test at topk=4. The proposer's hidden-state taps came from layers running topk=8; verification ran at topk=4 (or vice versa). The two distributions are not the same model.
3. **Prompt-formatting drift.** DFlash papers run with the model's chat template; the current bridge runs raw completion. Tokenization paths differ.
4. **One prompt is not a benchmark.** The Apr 24 result is statistically meaningless at n=1.

### The revival, three sub-tracks

**3.A — Calibrate the projector against IQ2_M target.**
- Generate a 200-prompt calibration set spanning factual / math / code / chat.
- For each prompt, capture target hidden states *at IQ2_M* and target logits at the topk we'll deploy at (topk=4).
- Re-fit the projector head (`fc.weight`, `hidden_norm.weight` in `war_room/dflash_mlx/python/projector.py`) by minimizing KL between proposer logits and target logits on the calibration set.
- This is a 200-MB fine-tune, runs in MLX, < 4 h on the M4 Air.

**3.B — Measure on a real distribution, not a single prompt.**
- 50-prompt evaluation set, separate from calibration.
- Report mean acceptance + 95% CI, not a single number.
- Acceptance threshold for "alive": **≥4/15 mean (≥27%)**. Below 4/15, dFlash is dead and Track 3 closes.

**3.C — If acceptance ≥ 4/15: ship a real end-to-end loop.**
- C++ multi-cycle generator: propose 16 → verify → roll back KV to last accepted → propose next 16. Currently the code only does one cycle.
- Target: net decode speedup ≥ 1.4× over Track 1 best, on a 256-token generation, quality gates pass.

### Important: dFlash and ANE are complementary, not competing
- ANE: speeds up the **target's** shared expert step.
- dFlash: amortizes **target verification** across multiple draft tokens per cycle.
- A target with ANE-accelerated shared experts is *also* a faster verifier. The two stack multiplicatively.

### Track 3 gate
3.A: projector recalibrated, KL on held-out set < 0.5.
3.B: mean acceptance ≥ 4/15 on 50-prompt eval. **If not, close Track 3 honestly.**
3.C: net 1.4× speedup over Track 1 best, quality gates pass.

---

## 8. Track 4 — TurboQuant, used for *quality*, not speed (3–5 days)

The April 6 measurements showed turbo3 KV gives only ~8 % memory reduction and negligible tok/s delta. As a **speed lever** TurboQuant is dead. As a **quality lever** it is alive: Lloyd-Max + WHT + QJL preserves attention dot-product fidelity better than naive int4. The point of Track 4 is to give the aggressive levers (topk=4, dFlash, ANE) more quality headroom.

**Steps:**
1. Run `llama-perplexity` on a 200-token slice of `wiki.test.raw` for: stock topk=8, topk=4, topk=4 + turbo3, topk=4 + q4_0. Establish the perplexity table.
2. **Verify:** topk=4 + turbo3 has perplexity ≤ stock topk=8 + 3 % (the master plan's hard gate).
3. **Use case 1:** when dFlash recalibration runs (3.A), the calibration loss is computed against the topk=4 + turbo3 target. The fine-tuned projector now matches the actual deployment config.
4. **Use case 2:** EvoDistill (Track 6) corpus generation runs at topk=4 + turbo3, so the fine-tune's adapter is shaped for the deployment quant.

TurboQuant becomes the **quality substrate** under everything else, not a feature in its own right.

### Track 4 gate
Perplexity table published. topk=4 + turbo3 within 3 % of stock topk=8. dFlash + EvoDistill both running on this substrate.

---

## 9. Track 5 — AutoResearch loop, Karpathy-style (parallel; ongoing)

Karpathy's recent talks have been describing a workflow where the agent itself runs the science: hypothesis → minimal experiment → metric → keep/discard → next hypothesis. This is what `evodistill/` was reaching for and what the kimi-overnight feedback file (in MEMORY.md) was a draft of. We make it real.

### What it is
A long-running orchestrator that:
1. Reads `PROGRESS.md` and the latest benchmark JSON.
2. Generates *one* concrete hypothesis ("Does increasing `--ub` from 128 to 192 at topk=4 raise decode > 1 %?")
3. Writes a one-shot bench script.
4. Runs it under the Track 0 reproducibility protocol.
5. Appends result to a structured ledger (`results/autoresearch_ledger.jsonl`).
6. If gain ≥ threshold and quality gates pass, commits a new daily driver and emits a Push notification.
7. Repeats.

### Why this is non-trivial
- The agent must not pollute its own context. Run as `nohup` background process; write all state to disk (the lesson from `feedback_overnight_context.md`).
- Each iteration `/compact`s. Inner loop is a `while true` over a Python runner, **not** an LLM agent loop.
- Hypothesis source: a queue file `autoresearch/hypotheses.yaml`, seeded with 30+ items by us, refilled by an LLM call only when empty (cheap).

### What goes in the seed queue (the seeds matter — they're the search space)
- Batch sweep: `--ub` ∈ {64, 96, 128, 192, 256}, `--b` ∈ {256, 512, 1024}.
- Threading: `-t` ∈ {2, 4, 6, 8}.
- KV quant: q4_0 vs turbo3 vs none, all ctx sizes.
- Routing strategy: topk=4 with experts ordered by router-logit-norm vs random.
- ANE dispatch (after Track 2): which subset of layers benefits — does it scale or saturate?
- dFlash (after Track 3): block size sweep 8 / 16 / 24 / 32.
- LM-head: top-P early exit threshold sweep.

### Connection to ANE and EvoDistill
- AutoResearch produces the **fitness gradient signal** that EvoDistill's ES optimizer uses to grade adapter candidates: an adapter is good if it (a) does not break perplexity, (b) raises decode tok/s on the standard suite, (c) produces *lower-entropy* router decisions (which makes ANE prediction and dFlash both win).
- AutoResearch produces the **dispatch decisions** that the ggml-coreml backend reads at runtime: which ops actually go to ANE based on measured benefit, not heuristic.

### Track 5 gate
Loop running unattended for 24 h, ≥10 hypotheses tested, ledger present, no context-overflow crash, 0 quality-gate regressions committed.

---

## 10. Track 6 — EvoDistill, ANE-shaped (overnight, in parallel; 1–2 weeks)

The `evodistill/` skeleton already exists with `e0_baseline_gate.py`, `e1_corpus_gen.py`, `e2_es_optimizer.py`. We finish it, with a twist that prior plans missed.

### The twist: hardware-aware fitness
The classical EvoDistill / EGGROLL fitness is `-cross_entropy(model + LoRA, corpus)`. We replace it with:

```
fitness = -CE(model+LoRA, corpus)
        + λ_route * (1 / router_entropy(model+LoRA))     # reward predictable routing
        + λ_share * shared_expert_activation_share        # reward more compute through static path
        + λ_speed * decode_tok_per_s(model+LoRA)          # measured, not estimated
```

with `λ_*` tuned so quality dominates but routing-shape contributes. The model is fine-tuned to be **easier for the ANE and easier for dFlash**, in addition to being smarter.

### Steps (per the master plan + this twist)
- E0: frozen baseline LiveCodeBench V6 + quality gate. Done if `evodistill/runs/baseline_frozen.json` exists; produce if not.
- E1: 500 prompts × 10 samples at T_train = 0.9. ~45 min on M4 Air.
- E2: 200 ES steps, population 16, sigma 0.01, target deeper 24 of 48 layers, **with the hardware-aware fitness above**.
- E3: temperature calibration sweep on val set.
- E4: 3 rounds of iterative loop, quality-gate at each round.
- E5: final LiveCodeBench V6 eval, must beat frozen by ≥ 3 pp.

### Track 6 gate
LiveCodeBench V6 pass@1 improves ≥ 3 pp over frozen, quality gates pass, **router entropy decreases vs frozen** (the hardware-shape signal), decode tok/s does not regress.

---

## 11. Track 7 — Innovations not in any prior plan

You asked me to think really thoroughly about something **no one has talked about**. I list eight candidates here, ranked, with the top three picked for execution and the others queued for AutoResearch (Track 5). The picked three are what makes this campaign novel beyond "execute the master plan harder."

### 7.A — *Heterogeneous Compute Trinity* — **PICK**
Most engines use one compute unit per forward pass. ANE *or* GPU *or* CPU. The M4's 120 GB/s bus is shared, so naively running all three contends for the same memory. **But staggered, they stream.**

- **GPU** runs routed experts (dynamic, IQ2_M, GPU-eligible).
- **ANE** runs shared experts + LayerNorms (static, FP16, ANE-eligible).
- **CPU (Accelerate / NEON)** runs token sampling, top-K routing, RoPE position update.

Today these run sequentially, gated by `waitUntilCompleted`. The innovation is a **dependency-graph scheduler** in `ggml_backend_sched` that issues GPU and ANE command buffers concurrently for the *same* layer (since their inputs are independent — both read `attn_output`), and the CPU runs the next-token sampling on the *previous* layer's logits while GPU+ANE are starting the next layer.

This is what Apple's CoreML does internally for vision models. It has not been done for an LLM forward pass on this stack. Hard part: residual sum at end of layer must wait on both — but if GPU+ANE finish within 5 % of each other (we tune by adjusting which sub-graph each owns), the wait is amortized.

Estimated gain: 25–40 %. This is the highest-ceiling innovation in the plan.

Implementation surface: `ggml/src/ggml-backend.cpp` scheduler + new `ggml-coreml.m` (from Track 2). Build on Track 2; Track 7.A starts when Track 2 closes.

### 7.B — *Routing-Predictive Prefetch via tiny ANE Router Twin* — **PICK**
The MoE router is `argmax(router_logits)` per token per layer. The decision blocks expert dispatch. **Innovation:** train a tiny (~5 M-param) **same-shape twin** of the router that runs on the ANE, asynchronously, **one token ahead**.

While GPU is computing token N's experts, ANE predicts token N+1's routing. By the time GPU starts N+1, the predicted experts are already prefetched and pinned. If prediction is wrong (small fraction), we fall back to standard routing — net cost is one wasted prefetch.

Why this beats DeepSeek's MoE prefetch: theirs uses last-token routing as a heuristic. Ours uses a *learned* model on the actual hidden state, running concurrently on a different compute unit. No serialization cost.

Estimated gain: 8–15 %, stacks with everything else.

Implementation: train the router twin via EvoDistill (Track 6) on captured hidden states. Compile to MLProgram. Wire into MoE dispatch in `ggml-metal.m`'s `mul_mat_id` path.

### 7.C — *Fused-decode kernel: LM-head + sample + embed + router-N+1 in one Metal dispatch* — **PICK**
Currently per-token decode runs **four serial dispatches**: LM-head matmul → CPU sample → embed lookup → next router. Each is ≥0.1 ms with sync; ~0.5–1 ms total per token wasted on dispatch overhead. At 25 t/s, that is 12–25 ms / s, i.e., 1.5–3 % wall-clock.

**Innovation:** one fused Metal kernel that:
1. computes LM-head logits in chunks with top-P early exit (Track 1.3),
2. runs `argmax` on the active chunk,
3. *gathers* the embedding row for the sampled token (just an indexed memcpy in unified memory),
4. computes router_logits for layer 0 using the freshly-embedded token,
5. emits the resulting hidden state ready for layer 1.

All in one `MTLComputeCommandEncoder`. No CPU round-trip until the next iteration's prefill needs it.

Estimated gain: 3–8 %. Not huge alone, but it removes dispatch latency from the critical path of every token, and it stacks with 7.A's overlap (CPU is now free to do work for the *previous* layer instead).

Implementation: new `metal/fused_decode.metal` + plumb into `llama_decode` in `vendor/anemll-flash-llama.cpp/src/llama.cpp`.

### Other candidates — queued for Track 5 (AutoResearch will explore)

**7.D — Predictive expert cache via per-prompt routing memo.** After ~50 tokens of generation the routing pattern stabilizes for many prompt types. Maintain a small per-conversation memo of "experts that fired in the last N tokens" and pin them. ~10 % gain on long generations.

**7.E — ANE pre-dequantization pipeline.** IQ2_M → FP16 unpack is a non-trivial GPU cost. ANE has spare cycles between shared-expert layers. Use ANE to dequant the *next* layer's IQ2_M weights into an FP16 staging buffer while GPU is running the current layer. Hides dequant entirely.

**7.F — Cross-conversation prefix KV cache with content-addressed lookup.** System prompts and common openings have identical KV. Hash → store → reuse across sessions. Free in unified memory; needs a small hash table.

**7.G — `madvise` tuning on the GGUF mmap.** Aggressive `MADV_WILLNEED` on next-likely expert pages. Internal NVMe is 3 GB/s; the unified-memory cache prefetcher does not know which expert pages will be hot. We do.

**7.H — MIL super-fusion across MoE layers.** Compile the entire decode loop's static sub-graph (norms + shared experts + residuals + KV update + LM-head core) as one MLProgram. The ANE compiler can fuse aggressively across layer boundaries when given a single graph. The reason no one has 0% → measurable ANE on llama.cpp is that everyone compiles per-op. We compile super-fused.

---

## 12. Sequencing — what runs when

```
Day 0 (today):              Track 0 — hygiene + reproducibility
Day 1–3:                    Track 1 — free wins (lookahead, top-P)
Day 1–3 (parallel):         Track 4 — perplexity table
Day 1–14 (parallel):        Track 5 — AutoResearch loop (always on)
Day 3–10:                   Track 2 — Phase 4 ANE shared experts
Day 3–14:                   Track 3 — dFlash revival
Day 7–14 (parallel):        Track 6 — EvoDistill with hardware-aware fitness
Day 10–21:                  Track 7.A — Heterogeneous Compute Trinity
Day 14–21:                  Track 7.B — Router twin
Day 21+:                    Track 7.C — Fused decode kernel
```

Tracks 5 and 6 run in the background continuously. Tracks 1, 2, 3 run as foreground engineering. Track 7 only starts once 2 has produced its `mactop` ANE > 0% screenshot.

---

## 13. Termination conditions

We do not stop until **all of these are true**:

1. Decode ≥ 45 tok/s on the standard suite, quality gates pass, 3-run median.
2. `mactop` ANE > 0 % sustained during decode for ≥ 30 s.
3. dFlash either ships ≥ 1.4× speedup or is documented closed with the projector-recalibration receipt.
4. EvoDistill's adapter merged delivers ≥ 3 pp on LiveCodeBench V6 over frozen.
5. AutoResearch ledger has ≥ 100 logged hypotheses with at least 5 commits to daily driver.

If we hit a wall before that — e.g., `mactop` refuses to show ANE > 0 even after Path B, or BW saturates at 38 t/s and overlap doesn't help — **document the wall with measurements**, not speculation, and call the campaign closed at the physical ceiling we found.

---

## 14. What never to do

- **Never** quote tok/s outside of the `bench_protocol.sh` 3-run median. Single-prompt single-run numbers are not engineering.
- **Never** trust an agent's summary over a benchmark JSON. Verify every claim by re-running the bench yourself.
- **Never** commit a daily-driver change that lowers any quality gate, even by one prompt.
- **Never** chase topk=1 or topk=2: they fail 23×17. Speed without correctness is fake speed.
- **Never** revive any AR-with-different-architecture-draft path. M-RoPE blocks it permanently — dead by physics, not engineering.
- **Never** copy Swift code from `references/SwiftLM-main/`. Port the algorithm to MSL.
- **Never** skip the Lisbon + 345 + FizzBuzz gate. Three prompts. They have caught every silent breakage in this project's history.
- **Never** make a claim that ANE is active without a `mactop` screenshot in the same commit.

---

## 15. Operating discipline (Karpathy + Apple Silicon ethos)

- **One bounded task at a time.** Never two concurrent code changes in the same commit.
- **Pre-commit rejection criteria.** Before any benchmark run, write down the metric that must improve and by how much, and the metric that must not regress. If the result violates it, revert.
- **Rejected experiments stay visible.** Append to `results/rejected.md`; never erase a failed path. Negative evidence is as load-bearing as positive.
- **Read actual files before forming opinions.** No assumed code structure. No hallucinated APIs.
- **Attribution mandatory.** Every algorithm cited (DFlash → Apple, EGGROLL → @rustane_dev, TurboQuant → DeepMind arXiv 2504.19874, Lookahead → Yifei Li 2024, AutoResearch loop → Karpathy 2026).
- **Local benchmark truth beats theory.** Your opinion, a paper, another agent's output: all subordinate to what the M4 measures.
- **No paper sections, no marketing, no announcements** until the benchmark exists and passes the protocol. The work first.

---

## 17. ADDENDUM 2026-04-25 — Findings from `papers:references:repos/` and `vendor/`

**TL;DR — Read this before anything else in §17:**

1. **dFlash is 85% done, not 0%.** `vendor/anemll-flash-llama.cpp/dflash/` is a 1,664-line C++ port of dFlash specifically configured for Qwen3.5-35B-A3B (target_layer_ids `{1,10,19,28,37}`). `war_room/execution_plan_v2.md` documents three specific bugs with file:line; **fix is ~1.5 h**, not 2 weeks. Realistic post-fix target: ≥38 t/s end-to-end, ≥60% acceptance.
2. **Path B (private MIL via `_ANEClient`) is empirically dead** on this M4. `ane_direct` ran here: 5 ANE classes resolve, but `_ANECompiler` rejects MIL with `InvalidMILProgram`. **CoreML is the only path to ANE** (Track 2). Drop Path B as fallback.
3. **MTL4 hardware works on M4** (`supportsTensors=YES`). llama.cpp gates it in software at `ggml-metal-device.m:701-708`. The +10–170% gain claim is from `MPSGraphExecutable`, NOT from flipping the gate. **New Track 11A** — public-API only, low risk, PR-shaped.
4. **ANE has its own bandwidth channels** (`aneDCSBytes`, `aneFabricBytes`) separate from `dramBytes`. The 120 GB/s "wall" in §1 is the GPU's wall; ANE adds a parallel channel. **Track 7.A is the load-bearing innovation** above ~30 t/s.
5. **TurboQuant's value is long context, not speed.** `vendor/llama-cpp-turboquant/` is in-tree; `build-turbo/bin/llama-cli` was just built. At 128K, FP16 KV is 16 GB (doesn't fit in 16 GB RAM); 2-bit TurboQuant KV is 2.1 GB (fits with 6 GB headroom). **New Track 10 — ANE Long-Context Engine** combines TurboQuant + ANE attention + dFlash + prefill chunking for 128K on 16 GB at ≥25 t/s, which is uncharted territory.
6. **`autoresearch-macos-master/` is a working fork target** for Track 5 — Karpathy's reference impl with macOS/MPS support. Fork it, swap `train.py` → `bench_inference.py`, keep the 5-min budget and TSV ledger.
7. **Models are present locally:** target `Qwen3.5-35B-A3B-UD-IQ2_M.gguf` and draft `Qwen3.5-35B-A3B-Draft-f16.gguf` at `/Users/manuelmonteiro/models/`. No download needed.
8. **silicon-\* skills installed** in `.claude/skills/`; probes compiled (`gpu_probe`, `soc_power`, `clpc_probe`, `intelligence_probe`, `ane_direct`). 41 system + 39 per-process bandwidth/energy/thermal channels are accessible without sudo on macOS 26.

**The single most impactful sentence in §17:** *the v3 "30→50 t/s research project" framing was wrong; the actual situation per `war_room/execution_plan_v2.md` is "31.6 t/s baseline + 1.5 h bug fix → ~38 t/s today, then layer on Tracks 11A/2/10 over the following weeks."*

---

The first-pass audit missed `papers:references:repos/` and the active state of `vendor/anemll-flash-llama.cpp/`. These findings **override prior assumptions** in §6, §7, §9 in specific ways. §17.17 is the current operational plan; earlier subsections are evidence and rationale.

### 17.1 — `apple-silicon-internals-main/` (the ANE reverse-engineering toolkit)
Already-compiled binaries: `ane_direct`, `gpu_probe`, `soc_power`. 10 Claude Code skills (`silicon-power`, `silicon-profile`, `silicon-watch`, `silicon-soc`, `silicon-bench`, `silicon-scan`, `silicon-xray`, `silicon-entitlements`, `silicon-ocr`, `silicon-detect`).

**Key facts that change the plan:**

1. **Their own conclusion (line 295 of their README):** *"Token generation at 19 tok/s is memory-bandwidth-bound (~100 GB/s DRAM). No software API change helps — you need either more bandwidth (M4 Pro/Max) or better quantization."* Tested:
   - `GGML_METAL_TENSOR_ENABLE=1` → no improvement.
   - MTL4 precompiled `MPSGraphExecutable` for Qwen-3B → only **1.08×**, for Qwen-4B → 1.10× (speedup *shrinks* with model size).
   - llama.cpp Metal shaders are 3× faster than MPSGraph.
   - BNNS attention is 0.003 ms — not the bottleneck.

   **Consequence:** §6 Track 2 expected gain is revised down to **+5–15%** from precompiled-graph dispatch alone. The real win is bandwidth-channel relief (§17.2 below), not raw speed.

2. **Path B (private MIL via `_ANEClient`) is fenced.** `pocs/ane_direct.m` proves MIL compilation fails at runtime without entitlements. Only `_ANEIOSurfaceObject` zero-copy I/O works. **Drop Path B from §6 Track 2.** CoreML is the only path. If CoreML refuses to schedule on ANE, debug it; there is no rescue path.

3. **GPU Tensor API is disabled on M4** (pre-M5/A19). Already confirmed in our own llama.cpp logs. Don't waste cycles trying.

### 17.2 — The actual ANE thesis: bandwidth-channel relief, not compute speed

`libIOReport.dylib` exposes per-subsystem channels: `dramBytes` (CPU/GPU shared port), `aneDCSBytes` (ANE's dedicated DCS port), `aneFabricBytes` (ANE fabric path). **AMC stats prove ANE has its own memory channels distinct from GPU's DRAM port.** ANE IOSurface I/O measured at 1479 GB/s.

**Implication:** The "120 GB/s wall" estimate in §1 is the *GPU's* wall. ANE adds a parallel channel. The realistic ceiling above 50 t/s is achievable IF compute can be sunk into ANE concurrently.

**Track 7.A (Heterogeneous Compute Trinity) is now the load-bearing innovation, not Track 2.** Track 2 is the *enablement* (a CoreML backend); the win comes from concurrent dispatch with measurable `dramBytes` going *down* per token while `aneDCSBytes` goes up.

### 17.3 — `dflash-mlx-main/` — DFlash's 89% acceptance is real on this exact model

Reference benchmarks (M5 Max 64GB, MLX 0.31.1):

| Target | Baseline | DFlash | Speedup | **Acceptance** |
|---|---|---|---|---|
| Qwen3.5-4B | 53.5 | 197.5 | 3.69× | 88.7% |
| Qwen3.5-9B | 31.1 | 127.5 | 4.10× | 89.0% |
| Qwen3.5-27B-4bit | 33.2 | 65.8 | 1.98× | 89.5% |
| **Qwen3.5-35B-A3B-4bit** | **140.0** | **242.9** | **1.74×** | **89.3%** |

**The Apr 24 sprint result of 0/15 was a config error, not a fundamental block.** Acceptance of ~89% is what's possible when target/draft are paired correctly:

- **Right target:** `mlx-community/Qwen3.5-35B-A3B-4bit` (NOT IQ2_M; the projector was trained against this distribution).
- **Right draft:** `z-lab/Qwen3.5-35B-A3B-DFlash` from HuggingFace.
- **Right runtime:** stock dflash-mlx with `RecurrentRollbackCache` (we don't have this in war_room — that's the missing piece).
- **Right architecture handling:** dFlash uses block-diffusion + tape-replay, which is **GatedDeltaNet-aware**. The M-RoPE block I documented as permanent in §2 was for AR speculative with **different-architecture drafts**, not for dFlash. **dFlash is not blocked by GDN/M-RoPE; it was designed for it.** Retract that claim.

**Strategic fork in the road for Track 3:**

| Path | Effort | Speedup | RAM | Quality |
|---|---|---|---|---|
| **3.X — Switch to MLX 4-bit** | 1–2 days | 1.74× proven | **17.5 GB → does NOT fit on 16GB M4 Air** | same as 35B |
| 3.Y — Switch to MLX with 9B-4bit | 1 day | 4.1× proven | ~5 GB, fits easily | smaller model |
| **3.Z — Port dflash-mlx kernels to llama.cpp IQ2_M** | 1–2 weeks | 1.4–1.7× projected | 10.6 GB, fits | 35B at IQ2_M (current) |

**Recommendation:** start with Path Y (smallest, validates the toolchain on this machine, ~9B-4bit at 30+ tok/s baseline → 120+ tok/s with dFlash, immediate proof). Then Path Z to keep the 35B quality with the proven kernel set. Path X is excluded by RAM constraint.

The dflash-mlx repo also contains:
- `dflash_mlx/recurrent_rollback_cache.py` — GDN-aware tape-replay (the war_room patches lack this; that's why our verification was broken).
- `dflash_mlx/kernels.py` — Metal kernels for tape replay + JIT SDPA 2-pass for `N >= 1024`.
- `dflash_mlx/runtime.py` — orchestration loop.

These are the **three files to read first** for Path Z.

### 17.4 — `turboquant_plus-main/` — TurboQuant is already integrated; pivot to long-context

Critical facts:
- **Our project's `vendor/llama-cpp-turboquant/` is a clone of this fork.** `ggml-turbo-quant.c` is in-tree.
- **`build-turbo/` exists in the project root.** The TurboQuant llama.cpp build is **already prepared**, possibly already built. (Verify: see §17.7 "actions to check first.")
- Quality (their data, wikitext-2 perplexity):
  - f16 → 6.121
  - q8_0 → 6.111 (baseline)
  - q4_0 → 6.142 (+0.5%)
  - **turbo3 → 6.194 (+1.4%)** — meets the master plan's 3% perplexity gate.
- Speed on M5 Max: turbo3 generation is 10.7 tok/s vs q8_0's 85.5 (8× gap from inverse WHT in the dequant path on every block).
- **Known fix exists:** "pre-rotate-queries" optimization (their issue #23) collapses rotation to a single per-query op, but needs GQA per-head reshape.

**The pivot in §8 Track 4 is correct and now sharper:** TurboQuant is not a speed lever, it is a **long-context quality substrate.**

`apple-silicon-internals-main/benchmarks/kvcache_compress.m` gives the punchline math for our exact constraint:

> **Qwen3.5-4B at 16 GB RAM:**
> - FP16 KV at 32K: 4.0 GB → total 8.2 GB
> - FP16 KV at 128K: **16 GB → DOES NOT FIT**
> - TurboQuant 2-bit KV at 128K: 2.1 GB → total 6.3 GB → **FITS**
> - TurboQuant 2-bit KV at 262K: 4.1 GB → total 8.3 GB → **FITS**

For **Qwen3.5-35B-A3B IQ2_M (10.6 GB weights)** the same math gives ~5 GB KV budget at FP16; with TurboQuant 2-bit, **128K context fits with headroom, 256K context is in reach**. Nobody else on Apple Silicon can do this on 16 GB.

### 17.5 — `autoresearch-macos-master/` — fork this, don't reinvent

Karpathy's 2026 reference impl, with a macOS/MPS fork. Three files:
- `prepare.py` — fixed (constants, data, eval). Read-only.
- `train.py` — the **single file the agent edits**. Architecture, optimizer, loop, hyperparams — all fair game.
- `program.md` — agent instructions ("kick off a new experiment", "loop forever, never stop").

5-minute wall-clock budget, ~12 experiments/hour, ~100 overnight. Output: TSV ledger with `commit, val_bpb, memory_gb, status, description`. Crashes get `status=crash`. Improvements get `keep`. Regressions get `discard` and a git reset.

Loop discipline is the key engineering content (§9 in the prior plan was wishful; this is concrete):
1. Look at git state.
2. Edit the one file.
3. `git commit`.
4. Run experiment, **redirect everything to log** (no tee, no context flood).
5. `grep` the metric out.
6. Crash → `tail` 50 lines, attempt fix or `discard`.
7. Improvement → keep the commit, advance branch.
8. Equal/worse → `git reset` to before.
9. **NEVER STOP.** Out of ideas → think harder, re-read papers, try more radical changes. Loop runs until human interrupts.

**The §9 Track 5 implementation is now: fork this repo, replace `train.py` with `bench_inference.py` (the file the agent edits to try llama.cpp flag combos / kernel patches), keep `prepare.py` for the bench protocol fixed-point, keep the 5-min budget, keep the TSV.** Don't write a new orchestrator.

**One important constraint from their `program.md`:** every iteration must redirect output to disk and only `grep` the result. This is the "context preservation" trick that makes overnight loops survive context overflow. It's also exactly what `feedback_overnight_context.md` in our memory file warned about.

### 17.6 — `mlx-flash-main/` — proven SSD streaming on this exact hardware

**Tested on M4 MacBook Air 16 GB with Nemotron-30B (17.8 GB on disk):**
- Normal mode: 4.1 s load, 18+ GB RSS (swap), laggy.
- Flash mode: **0.8 s load, 0.6 GB RSS, smooth.**

The mechanism (each idea is independently useful for this project):
1. **Trust the macOS unified page cache.** LRU eviction is faster than custom caches.
2. **Parallel `pread()`** (4–8 threads, GIL-releasing) for layer reads.
3. **`madvise(WILLNEED)`** to prefetch *next layer* while computing *current*.
4. **`madvise(DONTNEED)`** to release cold layers, keeping resident set 7–15 GB.
5. **FMA dequant on GPU** — Q4 → fp16 in one Metal kernel (no extra copies).
6. **MoE top-K streaming** — only top-K active experts read per token.

**Honest blocker they document:** prefill phase OOMs on long prompts because the host engine evaluates the full KV graph at once. Their Python monkey-patch can't yield memory back during prefill. **Solution requires C++ integration** — exactly the surface our llama.cpp fork can fix.

This is **Track 7.G in §11 made concrete** — the `madvise` + parallel `pread` for routed-expert prefetching. It is also the working reference for §17.8 below.

The lineage matters: Karpathy's `llama2.c` → Apple Research arXiv 2312.11514 *LLM in a Flash* (the PDF of which is at `papers:references:repos/2312.11514v3.pdf`) → `flash-moe` → `mlx-flash`. This is the canonical Apple Silicon LLM streaming chain.

### 17.7 — Actions to check before any new code

1. `ls build-turbo/bin/llama-cli` — does the TurboQuant build already produce a binary?
2. `du -sh build-turbo build-nsg-opt vendor/anemll-flash-llama.cpp/build* vendor/llama-cpp-turboquant/build*` — find which builds are real vs. duplicates.
3. `./build-turbo/bin/llama-cli ... --cache-type-k turbo3 --cache-type-v turbo3 ...` on the standard topk=4 baseline — measure turbo3 right now on this M4. The Apr 6 negligible-difference number used the build at that time; the current `vendor/llama-cpp-turboquant` may be newer.
4. Install the 10 silicon-* skills. Bake `silicon-watch` into `bench_protocol.sh`.
5. Run the existing `apple-silicon-internals-main/ane_direct` binary once. The output enumerates `_ANEModel`, `_ANEClient`, etc. methods in our installed macOS — confirms which APIs are present *now*, not in their test env.

### 17.8 — Revised innovation rank

The biggest new innovation surface, after these findings:

**🔥 Track 10 (NEW) — The ANE Long-Context Engine.** *Nobody on Earth runs 128K context on 16 GB Apple Silicon with a 35B-class model.*

Stack:
1. **TurboQuant 2-bit KV** (Track 4 substrate) → 128K cache fits in ~3 GB instead of ~16 GB.
2. **`madvise` + parallel `pread`** routed-expert streaming (from §17.6) → frees RAM headroom.
3. **At long context, attention compute (Q·Kᵀ scales O(n²)) starts to dominate** weight reads. Once KV is 2-bit and fits in ANE's working set, **route attention to ANE** via `ggml-coreml.m` (Track 2 enabler).
4. **dFlash on top** (Track 3) → multi-token verification per cycle hides ANE dispatch latency further.
5. **mlx-flash's prefill-chunking insight applied in C++** → solves the long-prompt OOM they documented. This is unique to a llama.cpp build.

**Result claim:** 128K context on 16 GB M4 Air at ≥25 t/s, with `mactop` showing ANE>0%. This is genuinely new territory and combines every reference repo into one delivery.

**Track 7.A (Heterogeneous Compute Trinity)** is upgraded from "innovation" to "load-bearing": after 17.2's bandwidth-channel finding, all gain above ~28 t/s short-context comes from concurrent GPU+ANE+CPU dispatch.

**Track 3 (dFlash) revival path is now explicit** — see 17.3. Path Y first (proves toolchain), Path Z next (keeps 35B at IQ2_M).

### 17.9 — Disk space asks (the user offered)

Suggested deletions to recover working space (please confirm before I act):
- `vendor/anemll-flash-llama.cpp/build/`, `vendor/anemll-flash-llama.cpp/build2/` if they exist (check size). Old builds.
- Time Machine duplicate dylibs in `build-nsg-opt/bin/` (`* 2.dylib`, `* 3.dylib`).
- `papers:references:repos/ml-ssd-main/` "* 2"-suffixed duplicate files (Time Machine clones).
- All `.DS_Store` files in `papers:references:repos/`.
- `results/` entries before Apr 21, archive to LaCie.
- `war_room/artifacts/*.bin` (200 KB × 3) and `war_room/logs/`, archive to LaCie.

What I would *not* delete: any of the reference repos in `papers:references:repos/`. After this audit they are load-bearing — every track depends on at least one of them.

### 17.10b — Empirical verification on this M4 (executed 2026-04-25)

The §17.7 verification list has been partially executed. Results:

**1. `ane_direct` — RAN on this M4. Confirms the §17.1 finding empirically.**
```
Classes resolved (all YES):
  _ANEInMemoryModelDescriptor, _ANEInMemoryModel, _ANERequest,
  _ANEIOSurfaceObject, _ANEDeviceInfo
precompiledModelChecksDisabled = NO
Compile: FAILED — Error: InvalidMILProgram (com.apple.appleneuralengine.compiler Code=1)
```
**Conclusion:** The 5 ANE private classes are accessible without entitlements on this exact macOS build. But MIL compilation of even a trivial `add(input, input)` graph fails. **Path B (private MIL via `_ANEClient`) is empirically dead on this machine.** This is not a documentation issue or a compile flag — the ANE compiler rejects our MIL syntax. CoreML's MIL compiler holds the entitlements that make MIL valid. **§6 Track 2 must use CoreML; Path B is permanently retired.**

**2. `soc_power` — RAN. Confirms the §17.2 channel topology empirically.**
- Enumerates **6 E-cores + 4 P-cores = 10 cores total** → confirms vanilla M4 (not Pro/Max).
- Per-core energy in mW (`ECPU0`–`ECPU5`, `PCPU0`–`PCPU3`), per-cluster (`ECPU`, `PCPU`, `ECPM`, `PCPM`), aggregate `CPU Energy`. Idle reading was 0.0 mW everywhere — the probe works, the system was idle.
- This is the per-subsystem attribution that all benchmarks must record from now on. The bench protocol in §4 step 5 must wrap inference invocations with `soc_power N 500` or `silicon-watch`.

**3. silicon-* skills — INSTALLED in `.claude/skills/`.**
All 10 skills copied and probes compiled (`clpc_probe`, `intelligence_probe` were source-only; built with `make probes/...`). The skills point at relative paths inside `apple-silicon-internals-main/`, so they must be invoked from that directory or with absolute paths — the next iteration of `bench_protocol.sh` will normalize this.

**4. `build-turbo/` — BROKEN, but recoverable.**
`build-turbo/CMakeCache.txt` exists with `CMAKE_BUILD_TYPE=Release`. Libraries built (`libllama.0.0.8448.dylib`, `libggml-metal.0.9.8.dylib`, etc.) but `bin/` contains only `llama-gemma3-cli`. The `llama-cli` and `llama-server` targets were never linked. **The fix is `cmake --build build-turbo --target llama-cli -j 4`** (currently running in background as task `b18301aby`). After it finishes, run the turbo3 baseline measurement.

**5. `dflash-mlx/kernels.py` and `recurrent_rollback_cache.py` — READ in full.**
The portability question for §17.3 Path Z is now answered:
- `tape_replay_kernel` is **~50 lines of Metal source** (lines 244–294 of `kernels.py`). It applies the recorded innovation tape to the snapshot KV state via per-head SIMD reduction. Direct port to ggml-metal: feasible in ~1 day.
- `batched_sdpa_2pass_exact` is **~150 lines of Metal source** (lines 479–565). Two-pass softmax with partials/sums/maxs reduction — same architecture as ggml-metal's flash-attention kernel; mainly a translation exercise.
- `RecurrentRollbackCache` is **~165 lines of Python** wrapping these kernels. The methods we need to mirror in C++: `arm_rollback`, `record_tape`, `rollback(n_accepted)`, `_rebuild_conv_state`. Estimated 1 week to port complete with tests.
- **Path Z effort revised down from "1–2 weeks" to "5–8 days"** with these references in hand.

**6. `inference_engine.m` — READ. Confirms the multi-backend reference impl.**
Three working paths in their bench:
- PATH 1 CPU via Accelerate `cblas_sgemm` (auto-dispatches to SME2 on M4).
- PATH 2 GPU via Metal compute shaders (`matmul`, `relu`, `linear_relu`).
- PATH 3 ANE via **IOSurface zero-copy I/O only** — they don't claim ANE compute, just I/O throughput (1479 GB/s in their README).

The hardware-feature detection block (lines 470–497) is gold — 5 lines of `sysctlbyname` calls give us SME, SME2, BF16, I8MM detection. **Track 0 step 4 (verified-reality table) should add this output to the per-machine fingerprint.**

### 17.11 — MAJOR FINDING: M4 Metal 4 ML Pipeline is hardware-supported, not blocked

This contradicts §17.1 finding #3 in this very document. I had asserted "GPU Tensor API is disabled on M4 (pre-M5/A19)" based on llama.cpp's runtime warning. **That's a software gate, not hardware.**

Empirical evidence from `apple-silicon-internals-main/`:
```
Device: Apple M4 (AGXG16GDevice)
supportsTensors: YES
supportsMachineLearningCommandEncoders: YES
```
The hardware advertises both. llama.cpp gates them in software at `ggml-metal-device.m:701-708`:
```objc
if (getenv("GGML_METAL_TENSOR_ENABLE") == NULL &&
    ![[dev->mtl_device name] containsString:@"M5"] &&
    ![[dev->mtl_device name] containsString:@"M6"] &&
    ![[dev->mtl_device name] containsString:@"A19"] &&
    ![[dev->mtl_device name] containsString:@"A20"]) {
    GGML_LOG_WARN("...tensor API disabled for pre-M5 and pre-A19 devices\n");
    dev->props.has_tensor = false;
}
```
With explanation comment (lines 696–700):
> *"M2 Ultra: ~5% slower; M4, M4 Max: no significant difference. TODO: try to update the tensor API kernels to at least match the simdgroup performance."*

**Reconciliation of the apparent contradiction with the apple-silicon-internals roadmap claim of "10–170% speedup":**

The +10–170% is **NOT from flipping the llama.cpp gate** (their own README explicitly says `GGML_METAL_TENSOR_ENABLE=1` gave no improvement on M4). It is from **switching to `MPSGraphExecutable`** — Apple's higher-level graph compiler — instead of using llama.cpp's hand-rolled simdgroup kernels at all.

Three distinct paths now exist for M4 acceleration, and they're additive:

| Path | What it is | Risk | Expected gain on M4 (decode) | Notes |
|---|---|---|---|---|
| **Track 2 — CoreML / `ggml-coreml.m`** | Compile MIL via CoreML; runtime auto-routes to ANE/GPU | Med (CoreML opacity) | +5–15% direct, more if ANE engages | Only path to ANE compute. Required for Track 10. |
| **Track 11A (NEW) — MPSGraphExecutable** | Compile graph via `MPSGraph`; run via `MTL4MachineLearningCommandEncoder` | Low (Apple-supported) | +10–170% per their data | Stays on GPU, uses optimized Apple kernels. |
| Track 11B — raw MTL4 tensor API | Use `MTLTensorDescriptor` + tensor ops directly via llama.cpp's existing path | High (impl gap) | "no significant difference" per llama.cpp authors | Skip. The kernels need rewriting first. |

**Track 11A is now elevated to top-tier innovation alongside Track 10.** It's the GPU-side acceleration that doesn't depend on ANE working. If Track 2's CoreML path stalls, Track 11A still ships gain. They're independent.

**Implementation sketch for Track 11A:**
1. Identify the hot ops: `mul_mat`, `mul_mat_id`, attention (`flash_attn_ext`), RMSNorm, RoPE, fused FFN.
2. For each, build an `MPSGraph` representation once at model load (use `MPSGraphCompilationDescriptor` with `optimizationLevel=.performance`).
3. Cache the compiled `MPSGraphExecutable` in `~/.cache/ane-turbostream/mps-graphs/<model-hash>/<op-name>.mpsgraphpackage`.
4. At inference time, encode via `MPSGraphExecutable.encodeToCommandBuffer:inputsArray:resultsArray:executionDescriptor:` instead of the simdgroup kernel path.
5. Validate numerical equivalence to within atol=1e-3 against current path on a 1K-token reference output.

**This is a clean PR-shaped contribution to llama.cpp upstream** if it works (the repo author's roadmap explicitly calls this out as Opção 1). Even if upstream rejects it, our fork carries it.

### 17.12 — The repo author's "powerinfer-mac" concept matches our project exactly

`docs/ROADMAP.md` proposes a product called **"powerinfer-mac"** with these features (item-by-item compared to my plan):

| Their proposal | Our track |
|---|---|
| 1. Hybrid ANE+GPU: prefill on GPU (batch parallel), decode on ANE (efficient) | **Track 7.A** Heterogeneous Compute Trinity |
| 2. Adaptive compute: switch backend based on IOReport (thermal, battery, throttle) | **Track 7.I** CLPC-aware adaptive backend |
| 3. Pre-compiled model cache: serialize MPSGraphExecutable to disk | **Track 11A** (above) |
| 4. Power budget mode: max perf when plugged, min consumption on battery | **Track 7.I** (subsumed) |

**They identified the exact same product gap I did, independently. This strongly validates the thesis.** They have the reverse-engineering toolkit; we have the implementation runway. Combining gives the first complete delivery.

The repo author's recommended order (their §"Recomendação"):
1. Blog post (we'll do this when something ships)
2. Open source the toolkit on GitHub (already there)
3. PRs to llama.cpp with MPSGraphExecutable path (= **our Track 11A**)
4. Build powerinfer-mac as product (= our entire plan)

We are **on the right path** by independent triangulation. The novel pieces we add beyond their roadmap:
- **Track 4** TurboQuant 2-bit KV (they don't have this).
- **Track 3** dFlash speculative decode (they don't have this).
- **Track 10** ANE Long-Context Engine (combines all three; nobody has this).
- **Track 5** AutoResearch overnight loop on llama.cpp configs (they don't have this).

### 17.13 — `bytes` channels for bandwidth attribution (consolidated reference)

From `docs/FINDINGS.md`, the per-subsystem channels available without sudo on macOS 26:

**System (41 metrics):**
- Power: `cpuPower`, `gpuPower`, `anePower`, `dramPower`, `displayPower`, `wifiPower`
- Thermal: `batteryTemperature`, `skinTemperature`, `thermalPressure`
- Energy: `cpuEnergy`, `gpuEnergy`, `gpuSRAMEnergy`, `aneEnergy`
- **Bandwidth: `dramBytes`, `aneDCSBytes`, `aneFabricBytes`** ← the load-bearing metrics for Track 7.A
- ANE-specific: `aneTime`
- Display: `displayFPS`, `edrHeadroom`

**Per-process (39 metrics):**
- Cost: `cpuCost`, `gpuCost`, `gpuTime`, `aneEnergy`, `aneTime`, `cpuInstructions`
- I/O: `bytesRead`, `bytesWritten`
- QoS breakdown: 7 levels
- Network: `networkCost`, `wifiIn`, `wifiOut`

**Bench protocol must record (per token, ideally):** `dramBytes` delta, `aneDCSBytes` delta, `aneFabricBytes` delta, `gpuPower`, `anePower`, `cpuPower`, `aneEnergy`, `gpuEnergy`, `cpuEnergy`, `thermalPressure`. The ratio `aneDCSBytes / dramBytes` is the **direct measurement of bandwidth-channel relief** that validates Track 7.A.

### 17.14 — Methodology pattern for future ANE/GPU spelunking

The repo's reverse-engineering methodology is a generic recipe. We should keep this in `docs/discovery_protocol.md` for the AutoResearch loop to reuse when it hits a closed-API wall:

```
1. dlopen() private framework from /System/Library/PrivateFrameworks/
2. objc_copyClassList() + class_getImageName() to find classes per framework
3. class_copyMethodList() / class_copyPropertyList() / class_copyIvarList()
4. objc_msgSend() dynamic dispatch to call discovered methods
5. dlsym() for C functions in private libraries (libIOReport.dylib pattern)
6. IOSurfaceCreate() for zero-copy I/O (the ANE pattern)
```

Their numbers: 2,152 frameworks scanned, 657+ ML/Intelligence classes, 93 Metal 4 classes, 34 ANE classes, 1009 IOReport channels. **The Track 5 AutoResearch loop should run this protocol nightly on any API surface that's not yet documented in our codebase.**

### 17.15 — DECISIVE FINDING: `vendor/anemll-flash-llama.cpp/` is the right base fork

The third llama.cpp fork in `vendor/` was the missing piece. It's not just Flash-MoE — it's a **superset of what we need**, already built and running on this M4:

**What it has (verified):**
- Working `llama-cli` binary at `vendor/anemll-flash-llama.cpp/build/bin/llama-cli` (5.2 MB Mach-O arm64). On startup: `MTLGPUFamilyApple9 (1009)` confirmed = our M4.
- A `dflash-cli` binary in the same `bin/` directory — **complete C++ port of dFlash for llama.cpp**.
- 88 `llama-*` binaries total including `llama-bench`, `llama-server`, `llama-perplexity`, `llama-speculative`, `llama-lookahead`, `export-graph-ops`.
- Slot-bank routed-expert streaming for MoE (`-DLLAMA_FLASH_MOE_GPU_BANK=ON` default), specifically debugged on Qwen3.5 GGUF MoE.
- Pinned to upstream llama.cpp commit `340807273b6aa765c9353804b7ce680920373cb6` (vendored 2026-03-29).

**What `dflash/dflash.h` reveals (verbatim from the header):**
- `dflash_params.target_layer_ids = {1, 10, 19, 28, 37}` — **exactly configured for Qwen3.5-35B-A3B's 40 layers**. This is not a coincidence; this fork was built for our model.
- `dflash_gdn_tape` struct — the GDN-aware rollback tape with `innovation_tape`, `tape_k`, `tape_g`, `tape_qkv`, `state_snapshot`, `conv_snapshot`. This is the C++ analog of `dflash-mlx`'s `RecurrentRollbackCache` — already done.
- `block_size=16` (matches dflash-mlx), `max_ctx=262144` (256K).
- API: `dflash_init`, `dflash_prefill` (captures target hidden states), `dflash_draft`, `dflash_verify`, `dflash_rollback`, `dflash_generate`, `dflash_acceptance_ratio`.

**The strategic implication is dramatic.**

What I had planned as "Track 3 Path Z — port dflash-mlx kernels to llama.cpp IQ2_M, 5–8 days" is **already done in `vendor/anemll-flash-llama.cpp/dflash/`.** The 2 months of `war_room/` patches may have been re-implementing what was already shipped here. We need to verify this works (next priority is to run `dflash-cli` against our IQ2_M model and measure acceptance) but if it does, we save 5–8 days.

**Updated fork strategy:**

| Fork | Status | Role going forward |
|---|---|---|
| `vendor/anemll-flash-llama.cpp/` | **NEW BASE** | Primary working tree. Already has dFlash + Flash-MoE. |
| `vendor/llama-cpp-turboquant/` | TurboQuant donor | Backport `ggml-turbo-quant.c` + Metal kernels into the new base. |
| `papers:references:repos/dflash-mlx-main/` | Reference for dFlash kernels | Compare C++ port vs MLX original; ensure 89% acceptance ports over. |
| `papers:references:repos/mlx-flash-main/` | Reference for SSD streaming | Compare slot-bank impl vs `madvise`+`pread` impl; pick best. |
| `build-turbo/` | TurboQuant baseline measurement only | Keep for reproducing the Apr 6 turbo3 numbers. |

**This collapses Tracks 3, 7.G, 12 into one base.** Tracks 4, 11A, 2, 10 become *additions* to anemll-flash, not separate codebases.

**Track 12 (NEW) — Verify and stabilize the existing dFlash C++ port.**
The first task on this base is to:
1. Run `dflash-cli` on `Qwen3.5-35B-A3B IQ2_M` with the 40-layer target_layer_ids preset.
2. Measure acceptance rate. The bar is 70%+ (the dflash-mlx Python ref hits 89% on the 4-bit MLX variant; IQ2_M will be lower but should still be productive).
3. If acceptance < 30%, debug the FC projection (`fc_weight` + `hidden_norm_weight` from draft model) — this is the single most likely cause of low acceptance, and it's the same thing that broke `war_room/`.
4. If acceptance > 50%, we have a working dFlash baseline today.

**Track 13 (NEW, replaces partial Track 7.G) — Audit slot-bank Flash-MoE on this exact model.**
The fork's anchor model is Qwen3.5 GGUF MoE. Our IQ2_M is one such MoE. Run `llama-cli -DLLAMA_FLASH_MOE_GPU_BANK=ON ...` and measure:
- Resident RAM with slot-bank vs without (`mactop` or `silicon-watch`).
- Token throughput vs the current baseline.
- SSD bandwidth to LaCie (the fork was bench-tested on internal NVMe; LaCie is Thunderbolt and may behave differently).

This may already be a +20–40% gain just by enabling the right flags on a build we already have.

### 17.16 — Revised "Day 0–4" plan (replaces parts of §17.10)

```
Day 0 (DONE):       silicon-* skills installed; ane_direct/soc_power working; references audited
Day 1 (NEXT):       Test dflash-cli with IQ2_M model. Measure acceptance. If >50%, dFlash is unblocked.
Day 1:              Test llama-cli with -DLLAMA_FLASH_MOE_GPU_BANK=ON; measure resident RAM + tok/s.
Day 1:              Run llama-bench from the anemll-flash build to establish the new baseline tok/s.
Day 1–2:            Backport ggml-turbo-quant.c + Metal kernels from llama-cpp-turboquant → anemll-flash.
Day 2–3:            Track 1 (lookahead, top-P) on the new anemll-flash baseline.
Day 2–4:            Fork autoresearch-macos as the experiment driver, point at anemll-flash builds.
Day 3–7:            Track 11A (MPSGraphExecutable) on the anemll-flash base.
Day 4–10:           Track 2 (ggml-coreml.m) on the anemll-flash base.
Day 7–14:           Track 7.A (heterogeneous trinity) — only meaningful once Track 2 lands.
Day 10–21:          Track 10 (ANE Long-Context Engine) — combines TurboQuant + ANE attn + dFlash.
```

The build-turbo/llama-cli rebuild started in §17.10b is **no longer on the critical path** — it's a sanity-check measurement, not the integration target. Let it finish in the background but don't block Day 1 work on it.

### 17.17 — RETRACTED. `war_room/` is the active integration workspace, not deletable.

I was wrong in the prior version of this section. After reading `war_room/execution_plan_v2.md` (dated 2026-04-23, 2 days ago), the actual state is:

- `vendor/anemll-flash-llama.cpp/dflash/` is **85% complete**, not done. It has three specific, documented bugs.
- `war_room/` is the **live fix-up workspace** for those bugs plus a Python/MLX hybrid runner.
- The team explicitly identified the bug list and put them on a 1.5-hour fix-rebuild-bench cycle.

**Verified baseline numbers from execution_plan_v2.md (this M4, 16 GB):**
| Metric | Value | Harness |
|---|---|---|
| Target decode IQ2_M, topk=8 (default) | 22.8 t/s | llama-bench |
| Target decode IQ2_M, topk=4 | 27.0 t/s | llama-bench |
| Target decode IQ2_M, topk=4 CLI | 31.6–34.7 t/s | llama-cli (interactive harness) |
| dflash-cli baseline (broken, no --moe-topk) | 2.80 t/s | dflash-cli --baseline |
| Draft forward (MLX, 16 tokens) | ~86 t/s | test_p1_kv.py |
| Draft+projector overhead | +117 ms | test_p1_kv.py |

**The three bugs from execution_plan_v2.md (verbatim, with file:line):**

1. **`dflash_verify` uses `llama_batch_get_one` (line 412)** — only enables logits for the LAST token. For verify-all, must use `llama_batch_init` and set `logits[i]=1` for all positions. *This is why acceptance silently broke in war_room's Apr 24 run (0/15).*
2. **`--dflash-draft` arg parsed but not used** — `dflash-cli.cpp` parses it but never passes to `dflash_init`. `draft_model_path` stays empty. Fix: wire CLI arg to `dflash_init` call in `run_dflash()`.
3. **`--moe-topk` not supported in dflash-cli** — explains why `dflash-cli --baseline` gives 2.80 t/s vs `llama-bench` 27.0 t/s. Fix: add `--moe-topk` CLI arg and pass as KV override.

**Three-track parallel strategy from execution_plan_v2.md:**
- Track A: C++ fix + rebuild (1.5h) — the three bugs above.
- Track B: Python/MLX hybrid runner (2h) — `war_room/dflash_mlx/python/hybrid_main.py`. Bypasses C++ bugs.
- Track C: Metal kernel optimization (deferred) — TurboQuant integration.

**Success criteria (from execution_plan_v2.md):**
| Metric | Target |
|---|---|
| Baseline w/ topk=4 | ≥ 25 t/s |
| DFlash-AR acceptance | ≥ 60% |
| DFlash-block acceptance | ≥ 80% |
| Overall tok/s | ≥ 38 t/s |
| Draft cycle latency | ≤ 200 ms |

**The 38 t/s target is realistic given the dflash-mlx 1.74× speedup.** 27 t/s baseline × 1.4× (degraded acceptance on IQ2_M vs MLX 4-bit) = ~38 t/s.

**This rewrites the day-by-day plan radically:**

```
Day 0 (DONE):       silicon-* skills installed; ane_direct/soc_power/probes verified
Day 0 (NEXT 1-2h):  Apply Track A fixes (3 bugs), rebuild dflash-cli
Day 0 (next 30m):   Run dflash-cli --baseline with --moe-topk=4. Confirm ≥25 t/s.
Day 0 (next 30m):   Run dflash-cli --dflash-ar with draft model. Measure acceptance.
Day 1 (if needed):  If acceptance <50%, debug the FC projection / hidden state capture path.
                    The hybrid_main.py Python runner exists as fallback to localize the bug.
Day 1–2:            Once dFlash-AR ≥60%, layer on Track 1 free wins (lookahead, top-P).
Day 2–4:            Backport ggml-turbo-quant.c from llama-cpp-turboquant → anemll-flash.
Day 2–4:            Fork autoresearch-macos (Track 5).
Day 4–7:            Track 11A MPSGraphExecutable on the working dFlash baseline.
Day 7–14:           Track 2 (ggml-coreml.m / ANE).
Day 14+:            Track 7.A (heterogeneous trinity), Track 10 (long-context engine).
```

**Models present and ready:**
- Target: `/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf`
- Draft: `/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-Draft-f16.gguf` (also `-fixed.gguf` variant)

**What we keep from `war_room/`:**
- `war_room/execution_plan_v2.md` — the authoritative bug list and success criteria.
- `war_room/dflash_mlx/python/hybrid_main.py` — the Python/MLX fallback for bug localization.
- Any analysis notes from the Apr 24 0/15 run that documented WHICH part of the pipeline produced the 0% acceptance.
- The benchmark protocol scripts and prompt sets.

**What needs to come out of `war_room/` and into the main repo:**
- The three Track-A bug fixes once verified — they belong in `vendor/anemll-flash-llama.cpp/dflash/` directly, not as patches in war_room.
- The hybrid_main.py once it shows the bug localization works — should become a real `tools/dflash-debug.py` or similar.

---

## 18. First action when you (the agent) wake up

1. Read this document end to end. **§17 supersedes earlier sections where they conflict; §17.17 is the most current operational plan.**
2. Read `PROGRESS.md`, `war_room/execution_plan_v2.md`, and the verified-reality table in §2.
3. **The §17.7 verification list is DONE** (see §17.10b). Skip it. silicon-* skills are installed in `.claude/skills/`; probes compiled; `ane_direct` empirically confirmed Path B is dead; `soc_power` works without sudo; `build-turbo/bin/llama-cli` is built.
4. **Execute Track A from `war_room/execution_plan_v2.md`** — three documented bugs in `vendor/anemll-flash-llama.cpp/dflash/`, ~1.5 h fix-rebuild-bench cycle. This is the highest-leverage hour of work in the entire plan.
5. Bench `dflash-cli` with the local Qwen GGUFs (target IQ2_M + draft f16, both at `/Users/manuelmonteiro/models/`). Record the §17.13 channels per token. Compare to §17.17 success criteria (≥25 t/s baseline, ≥60% DFlash-AR acceptance, ≥38 t/s end-to-end).
6. Only after Track A is measured: layer on Track 1 free wins, then Track 11A (MPSGraph), then Track 5 (AutoResearch loop), then Tracks 2/7.A/10 per §17.17.

You are not in a hurry. You are in a campaign. The M4 will still be here tomorrow. The numbers either reproduce or they don't. We do not stop until we hit the wall.

— Apple Silicon AI Engineering Team, 2026-04-25
