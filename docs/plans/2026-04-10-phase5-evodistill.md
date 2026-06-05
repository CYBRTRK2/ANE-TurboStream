# Phase 5: EvoDistill — Gradient-Free On-Device Fine-Tuning

**Date:** 2026-04-10
**Phase:** 5 of Master Plan
**Target:** Better model quality (pass@1 on LiveCodeBench V6 > frozen baseline by ≥5pp)
**Prerequisite:** Working inference engine at 26 tok/s (DONE)

---

## Phase Status Summary (Phases 0-4)

| Phase | Name | Target | Result | Notes |
|-------|------|--------|--------|-------|
| 0 | Foundation | 7-10 tok/s | DONE: 25.5 tok/s | Stock llama.cpp anemll-flash fork |
| 1 | SSD Streaming | 15-25 tok/s | FAILED: OOM on 16GB | Slot-bank 32 requires >16GB RAM |
| 2 | TurboQuant KV | 18-28 tok/s | DONE: 25.88 tok/s | turbo3 KV, 99.4% parity, 8% less GPU mem |
| 3 | GPU Optimizations | 22-32 tok/s | DONE: 26 tok/s ceiling | All ops memory-BW bound, GEMV/ANE can't help |
| 4 | ANE Integration | 30-40 tok/s | FAILED: ANE slower than CPU | GEMV unfavorable on ANE, shared 120GB/s bus |

## Phase 5 Implementation Plan

Per Master Plan, Phase 5 uses:
- **SSD (Self-Distillation)** — Apple Research (arXiv 2604.01193) — training signal
- **EGGROLL** — @rustane_dev (March 2026) — ES+LoRA gradient-free optimizer
- **LiveCodeBench V6** — Apple Research ml-ssd evaluation harness
- **ANE for forward passes** — CoreML model compilation for all ES fitness evaluations

### Attribution (Donor Repos)
- `papers:references:repos/ml-ssd-main/` — LiveCodeBench V6 eval harness (Apple Research)
- `papers:references:repos/autoresearch-macos-master/` — Autonomous research loop methodology
- `papers:references:repos/apple-silicon-internals-main/` — ANE private API access (_ANEInMemoryModel)
- TurboQuant algorithm — `papers:references:repos/turboquant_plus-main/` (WHT rotation, QJL)

### Steps (from Master Plan)

- **E0**: Infrastructure setup (pip install mlx/mlx-lm/torch, set up evodistill/ directory, baseline eval)
- **E1**: Corpus generation (500 prompts × 10 samples = 5000 forward passes at T_train=0.9)
- **E2**: ES optimizer (EGGROLL: 811K LoRA params, antithetic ES, P=16, S=200 steps, deeper 24/48 layers)
- **E3**: Temperature calibration (sweep T* on held-out val set)
- **E4**: Iterative loop (K=3 rounds: corpus → ES → calibrate → quality gate → merge)
- **E5**: LiveCodeBench V6 final evaluation

### Key Architecture Decisions

1. **ANE for forward passes**: Use CoreML to compile Qwen3.5-35B-A3B (or LoRA-adapted version) to ANE for all ES fitness evaluations. This is the "ANE utilization >0%" gate from Phase 4, adapted to the fine-tuning use case where batch inference is needed.

2. **Autoresearch loop**: Following autoresearch-macos-master's methodology — bounded runtime experiments, keep/discard/crash ledger, branch advancement only on measured gains, autonomous iteration.

3. **No backward passes**: EGGROLL's ES optimizer needs only forward passes to estimate gradients, making it fully compatible with ANE and on-device execution.

4. **Quality gate remains**: Lisbon + 345 + FizzBuzz before every advance, plus val_loss tracking.