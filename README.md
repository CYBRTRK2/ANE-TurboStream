# ANE-TurboStream Clean Restart

This repository is a clean-room restart of the ANE-TurboStream effort.

We explicitly archived the previous mixed-runtime repository and restarted from scratch with one foundation only:
- llama.cpp
- specifically the Anemll Flash-MoE fork vendored at `vendor/anemll-flash-llama.cpp`

The sole strategic source of truth is:
- `docs/foundation/ANE TurboStream Master Plan.docx`
- `docs/foundation/ANE TurboStream Master Plan.extracted.txt`

## Archive

The previous repository snapshot was archived intact at:
- `/Users/manuelmonteiro/Desktop/ANE project_archive_20260404_134904`

Nothing from that archive is part of the active implementation baseline unless it is deliberately re-imported after measurement and review.

## Active rules

1. Local benchmark truth beats theory.
2. No reuse of old runtime code by default.
3. llama.cpp is the foundation.
4. Every optimization must have:
   - exact command
   - exact benchmark artifact
   - exact acceptance / rejection criteria
5. No claims of ANE acceleration without measured proof.

## Repository layout

- `docs/foundation/` — master plan and immutable planning artifacts
- `docs/plans/` — executable implementation plans for the clean restart
- `scripts/` — reproducible build / benchmark helpers
- `benchmarks/` — benchmark definitions and run manifests
- `results/` — local output artifacts
- `vendor/anemll-flash-llama.cpp/` — fresh clean clone of the llama.cpp foundation

## Immediate goal

Phase 0 is to establish a fresh defended llama.cpp baseline on this machine for:
- model: `Qwen3.5-35B-A3B-UD-IQ2_M.gguf`
- hardware: M4 MacBook Air 16 GB
- backend: Metal
- runtime: Flash-MoE llama.cpp fork

We do not promote any architectural ideas before that baseline exists in the new clean workspace.

The local donor corpus under `papers:references:repos/` is part of the active research loop. We mine it aggressively for ideas, measurements, and workflows, but every imported idea must earn its place with new local benchmarks in this clean workspace.
