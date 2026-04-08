# Stock-First KV Optimization Plan

Goal: continue the clean-room campaign on the strongest current path: stock llama.cpp with stock-compatible KV cache optimization, using TurboQuant+ as a donor rather than a blind transplant.

Architecture: keep stock mode as the mainline serving path because it is already strong on this machine. Treat slot-bank as a research lane. Build the next optimization phase around KV cache compression and longer-context efficiency, starting with built-in cache types and then auditing the minimum TurboQuant port surface.

Tech stack: anemll-flash-llama.cpp, Metal, GGUF Qwen3.5-35B-A3B, built-in cache types (`q8_0`, `q4_0`), TurboQuant+ donor.

---

## Task 1: Preserve current evidence

Objective: freeze the slot-bank pivot decision and current stock baseline.

Files:
- Keep: `results/p0_summary.md`
- Keep: `results/p4_summary.md`
- Keep: `results/p5_summary.md`

Verification:
- these summaries exist and are referenced in future work

## Task 2: Build a reliable stock KV test harness

Objective: stop relying on `llama-bench` for cache-type tests if it cannot create context in this fork.

Files:
- Create: `scripts/stock_kv_probe.sh`
- Output: `results/stock_kv_*.txt`

Requirements:
- support baseline, q8_0, q4_0
- support `--reasoning off`
- log exact command, prompt throughput, generation throughput, and memory breakdown lines

Verification:
- one baseline probe
- one q8_0 probe
- one q4_0 probe
- all produce readable artifacts

## Task 3: Measure context-scaling behavior

Objective: determine whether built-in KV compression helps as context grows on this machine.

Configs to probe:
- baseline f16 cache
- q8_0 cache
- q4_0 cache

At least these context sizes:
- 2048
- 8192
- 16384 if it fits

Acceptance:
- capture whether compressed KV improves fit or throughput at larger context
- if no measurable benefit, log it honestly

## Task 4: Audit TurboQuant patch surface locally

Objective: map donor files to local fork files before writing any code.

Donor source:
- `papers:references:repos/turboquant_plus-main/README.md`
- donor file list in Phase 5 summary

Local target surfaces:
- `vendor/anemll-flash-llama.cpp/ggml/include/ggml.h`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-common.h`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-quants.h`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml.c`
- `vendor/anemll-flash-llama.cpp/ggml/src/CMakeLists.txt`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
- `vendor/anemll-flash-llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m`
- `vendor/anemll-flash-llama.cpp/common/arg.cpp`

Deliverable:
- a local file-to-file patch map
- explicit list of what is algorithmic, what is CLI plumbing, and what is Metal backend work

## Task 5: Pre-commit the first TurboQuant experiment goal

Objective: define the success criteria before porting anything.

Primary success criteria:
- correctness first:
  - prompt still answers Lisbon correctly
  - no model-load failures
  - no obvious quality break on short smoke prompts
- memory/context second:
  - lower KV footprint or larger context fit than current stock cache baseline
- speed third:
  - do not assume speedup; require measurement

Rejection criteria:
- if the first port only reproduces the donor’s large Apple-Silicon slowdown, reject it as a mainline speed optimization and keep it as a research branch only

## Decision rule

Mainline stays on stock mode until a KV-compression patch proves itself on this machine with local evidence.
