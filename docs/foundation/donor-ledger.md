# Donor Ledger

This file maps donor assets to concrete use in the clean-room ANE-TurboStream restart.

## Runtime foundation

### vendor/anemll-flash-llama.cpp
- role: active implementation base
- use now: build, benchmark, sidecar, slot-bank, oracle, bank modeling
- remeasure required: yes, always local

## Sparse streaming donors

### papers:references:repos/flash-moe-main
- role: historical MoE streaming donor
- importable ideas:
  - deferred GPU CMD3 pattern
  - pread scheduling / page-cache lessons
  - rejection heuristics from prior experiments
- caution:
  - custom runtime only, not active base
  - do not transplant blindly

### papers:references:repos/mac-code-main/research/flash-streaming
- role: dense out-of-core streaming donor
- importable ideas:
  - F_NOCACHE direct I/O
  - weight split by access pattern
  - proof artifacts for what batching does not fix
- caution:
  - dense-model streaming economics are much worse than MoE slot-bank economics

## KV compression donors

### papers:references:repos/turboquant_plus-main
- role: KV compression algorithm donor
- importable ideas:
  - WHT rotation
  - QJL path
  - codebook / codeword layout
  - perplexity validation methodology
- caution:
  - README shows strong compression quality but severe speed regression on M5 Max due to inverse rotation overhead
  - speed path must be redesigned, not copied naively

### papers:references:repos/mac-code-main
- role: practical q4_0 KV-cache deployment heuristics
- importable ideas:
  - operational use of llama.cpp cache-type-k/v flags
  - long-context serving recipes
- caution:
  - headline speed claims must be revalidated locally

## Apple/ANE donors

### papers:references:repos/apple-silicon-internals-main
- role: Apple Silicon telemetry and private framework reconnaissance donor
- importable ideas:
  - SoC telemetry tools
  - power / subsystem measurement
  - framework scanning for ANE-related APIs
- caution:
  - not proof of a production-ready ANE inference path for this repo
  - use to inform measurement and feasibility, not to skip benchmarks

## Autonomous research loop donor

### papers:references:repos/autoresearch-macos-master
- role: experimental control-loop donor
- importable ideas:
  - baseline-first run
  - fixed-time budget experiments
  - keep/discard/crash ledger
  - branch advancement only on measured gains
  - autonomous overnight iteration without asking the user to continue
- adaptation target:
  - inference optimization loop: one bounded runtime change -> benchmark -> keep/discard/crash -> champion advances only on measured gains
  - Phase 4/5 forward-only self-distillation and LoRA/ES loops
  - future EvoDistill loop: corpus generation -> forward-only update -> eval -> champion retention
  - also useful for inference optimization campaigns where each run has a bounded benchmark script and automatic accept/reject logic

## Eval donor

### papers:references:repos/ml-ssd-main
- role: evaluation and future self-distillation donor
- importable ideas:
  - LiveCodeBench V6 harness
  - SSD-style evaluation workflow
- caution:
  - inference engine must be stable before this becomes active work

## Stub / binary-only donor

### papers:references:repos/flash-moe-m5-nax
- current contents observed:
  - `ppl_tokens.bin`
  - `ppl_tokens_2k.bin`
- role:
  - inspectable artifacts only for now
- action:
  - if more files appear later, reclassify
