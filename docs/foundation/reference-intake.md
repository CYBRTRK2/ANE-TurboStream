# Donor Reference Intake

This clean-room restart will actively use the local donor corpus at:
- `papers:references:repos/`

These materials are authoritative for ideas, measurement design, API reconnaissance, and algorithm porting, but not for blind code reuse. Every imported idea must be re-measured in the new llama.cpp workspace.

## Priority donor map

### 1. `vendor/anemll-flash-llama.cpp`
Role:
- primary runtime foundation
- slot-bank, sidecar, oracle, bank-modeling, Metal llama.cpp path

Use for:
- Phase 0 stock baseline
- Phase 1 slot-bank 32
- oracle-all-hit / oracle-prefetch ceilings
- future clean benchmark harnesses

### 2. `papers:references:repos/flash-moe-main`
Role:
- historical sparse-streaming / Metal MoE performance ideas

Use for:
- pread scheduling ideas
- deferred GPU compute pipeline ideas
- page-cache lessons
- reject/keep heuristic inspiration

Do not use as active runtime base.

### 3. `papers:references:repos/turboquant_plus-main`
Role:
- KV compression donor

Use for:
- WHT / codebook / QJL math
- perplexity validation methodology
- cache-type integration study
- identifying which pieces are portable into the anemll llama.cpp fork

### 4. `papers:references:repos/apple-silicon-internals-main`
Role:
- Apple Silicon / ANE / private framework reconnaissance

Use for:
- power and telemetry measurement
- private framework scanning
- ANE / Metal / system-level constraints awareness
- evidence collection for later ANE phases

Do not treat as proof of production-ready ANE acceleration for this project.

### 5. `papers:references:repos/ml-ssd-main`
Role:
- evaluation and future forward-only fine-tuning reference

Use for:
- LiveCodeBench evaluation harness
- future EvoDistill/SSD planning after inference foundation is stable

### 6. `papers:references:repos/mac-code-main`
Role:
- systems ideas donor for local agents, KV cache strategy, and flash-streaming experiments

Use for:
- llama.cpp serving flags and practical Apple-Silicon run recipes
- q4_0 KV-cache operational heuristics
- MLX persistent-context ideas worth comparing against llama.cpp direction
- flash-streaming findings, especially what failed and why

### 7. `papers:references:repos/autoresearch-macos-master`
Role:
- autonomous experiment-loop donor

Use for:
- keep/discard experiment protocol
- fixed-budget autonomous iteration
- result logging discipline
- branch-and-advance research loop for future forward-only training and performance search

### 8. `papers:references:repos/flash-moe-m5-nax`
Role:
- artifact stub / binary evidence only

Use for:
- inspect any stored token/perplexity artifacts if needed

Current note:
- this directory currently appears to contain only `ppl_tokens.bin` and `ppl_tokens_2k.bin`, not a full documented codebase

## Working rule

For every optimization proposal, log:
- donor source file(s)
- what idea is being imported
- what is being changed in llama.cpp
- benchmark acceptance / rejection criteria

## Immediate intake tasks

1. Read anemll Flash-MoE README and bank-modeling docs.
2. Read Flash-MoE README/paper for sparse streaming heuristics.
3. Read TurboQuant+ README and test layout for algorithm surfaces.
4. Read apple-silicon-internals README and benchmark/probe tooling.
5. Create a living donor ledger before implementation work begins.
