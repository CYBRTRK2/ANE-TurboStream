# Clean-Room llama.cpp Restart Plan

> For Hermes: execute this plan in the new clean workspace only. Do not pull implementation code from the archived repo unless explicitly approved after evidence review.

Goal: restart ANE-TurboStream from a fresh llama.cpp foundation and rebuild the project in a controlled, benchmark-first way while actively mining the local donor corpus in `papers:references:repos/` for ideas, algorithms, telemetry tooling, and validation workflows.

Architecture: use the Anemll Flash-MoE llama.cpp fork as the only runtime base. Treat the archived repository as historical evidence only. Reintroduce ideas only after reproducing them against a clean baseline in this new workspace.

Tech stack: CMake, Metal, llama.cpp, Flash-MoE sidecar tooling, benchmark scripts, GGUF Qwen3.5-35B-A3B, and the local donor corpus under `papers:references:repos/`.

---

## Phase A: Clean baseline

### Task 1: Verify local inputs
Objective: confirm model path and workspace prerequisites.

Files:
- Read: `docs/foundation/ANE TurboStream Master Plan.extracted.txt`
- Read: `vendor/anemll-flash-llama.cpp/README.md`
- Verify model: `/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf`

Acceptance:
- plan present
- vendor repo cloned
- model file present

### Task 2: Configure and build llama.cpp
Objective: produce fresh binaries in the clean workspace.

Files:
- Use vendor tree only: `vendor/anemll-flash-llama.cpp/`
- Output build dir: `vendor/anemll-flash-llama.cpp/build/`

Command:
- `cmake -S vendor/anemll-flash-llama.cpp -B vendor/anemll-flash-llama.cpp/build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_FLASH_MOE_GPU_BANK=ON`
- `cmake --build vendor/anemll-flash-llama.cpp/build --config Release -j$(sysctl -n hw.ncpu) --target llama-cli llama-bench llama-perplexity`

Acceptance:
- `llama-cli`, `llama-bench`, and `llama-perplexity` exist in `build/bin/`

### Task 3: Run fresh Phase 0 baseline
Objective: establish the first defended clean-room measurement.

Script:
- `scripts/phase0_baseline.sh`

Outputs:
- `results/p0_stock_baseline.md`
- `results/p0_stock_smoke.txt`
- `results/p0_stock_notes.txt`

Acceptance:
- one successful stock benchmark
- one smoke generation
- exact command logged
- no performance claims until repeated run confirms stability

## Phase B: Slot-bank reality check

### Task 4: Extract sidecar from clean workspace
Objective: produce routed-expert sidecar with the vendor tooling only.

Outputs:
- `results/sidecar/`
- `results/sidecar_extract.log`

Acceptance:
- extraction completes
- sidecar verifies cleanly

### Task 5: Measure slot-bank and oracle ceilings
Objective: separate hit-path limits from miss-path limits.

Runs required:
- slot-bank 32
- oracle-all-hit
- oracle-prefetch

Acceptance:
- three benchmark artifacts with exact commands
- decision memo: is sparse miss path still a first-order bottleneck or not?

## Phase C: Re-entry gates for innovation

No ANE work, TurboQuant work, or LM-head redesign is allowed into active implementation until:
- fresh stock baseline exists
- slot-bank 32 exists
- oracle-all-hit exists
- oracle-prefetch exists
- repeated-run variance is understood

## Decision rule

Any idea from the archived repository may be reconsidered only if:
1. it addresses a measured bottleneck in the clean workspace
2. it has a bounded benchmark plan
3. it has pre-committed rejection criteria
4. it beats the clean baseline on this machine
