# ANE-TurboStream

> **Running Qwen3.5-35B on a 16 GB MacBook Air — and pushing it faster.**

Research project exploring the physical limits of LLM inference on Apple Silicon.  
Target: Qwen3.5-35B-A3B (IQ2_M, 10.6 GB on disk) on an Apple M4 MacBook Air with 16 GB unified memory.  
Baseline: **26.02 tok/s** (April 2026). Goal: break 50 tok/s through a stack of compounding optimisations.

---

## Why this is interesting

Apple Silicon has one unified memory pool shared by CPU, GPU, and Neural Engine (ANE).  
A 35B MoE model at IQ2_M quantisation just barely fits in 16 GB, leaving almost no headroom.  
Standard wisdom says you're memory-bandwidth-bound at ~120 GB/s and there's nothing left to do.

This project is a structured attempt to prove that wrong — through speculative decoding, KV cache compression, ANE heterogeneous dispatch, and autonomous experimentation.

---

## Hardware & model

| | |
|---|---|
| **Hardware** | Apple M4 MacBook Air, 16 GB unified memory |
| **SoC** | M4 (6E+4P cores, GPU, ANE — separate bandwidth channels) |
| **Model** | `Qwen3.5-35B-A3B-UD-IQ2_M.gguf` |
| **Params** | 35B total / ~3B active per token (Mixture of Experts) |
| **Size on disk** | 10.60 GiB |
| **Inference engine** | llama.cpp (anemll-flash-llama.cpp fork) |

---

## Measured results

All numbers are reproducible via `scripts/bench_protocol.sh`.

### Official baseline — April 25 2026

| Config | Prefill pp512 | Decode tg128 | vs. stock |
|---|---:|---:|---|
| stock topk=8 | 283.18 ± 1.37 t/s | 23.77 ± 0.42 t/s | baseline |
| **stock topk=4** | **340.81 ± 1.73 t/s** | **26.02 ± 0.25 t/s** | **+9.5%** |
| shared-only (no routed experts) | 539.70 ± 0.23 t/s | 32.89 ± 0.53 t/s | +38.4% |

> `topk=4` is the official baseline: full quality, verified on reasoning tasks.
> The gap between 26 and 33 t/s (shared-only) is the research target — closing it means successfully offloading routed-expert compute to ANE or speculative decoding.

### Track results to date

| Track | Status | Finding |
|---|---|---|
| Track 0 — Reproducibility | ✅ Done | `bench_protocol.sh` locked; silicon-* skills installed |
| Track 1 — Speculative (`llama-lookup`/lookahead) | ❌ Rejected | 0% acceptance on recurrent/GDN architecture |
| Track 2 — CoreML/ANE backend scaffold | 🔄 In progress | `libggml-coreml` links; ANE device exposed via `GGML_COREML_ENABLE=1` |
| Track 3 — DFlash speculative decoding | 🔄 85% done | C++ port builds and loads both GGUFs; one-cycle probe passes |
| Track 4 — TurboQuant KV compression | 🔄 Validated | 4.9× compression, +1.4% PPL vs q8_0; speed optimisation in progress |
| Track 5 — AutoResearch loop | ✅ Running | Overnight flag sweeps; TSV ledger in `results/autoresearch/ledger.tsv` |
| Track 11A — MPSGraphExecutable | 🔧 Planned | M4 hardware confirmed capable; llama.cpp CMake gate identified |

---

## Novel contributions

### `turbo_metal_port/` — Metal shaders for TurboQuant KV cache
Six `.metal` files porting the TurboQuant algorithm ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874), ICLR 2026) to run on Apple Silicon GPU:
- Walsh-Hadamard Transform constants and dequant kernel
- PolarQuant set-rows kernels
- Flash-attention extension operating on TurboQuant-compressed keys/values

Enables **4.9× KV cache compression at under 1.5% perplexity cost** — making 128K context possible on a 16 GB Mac where standard FP16 KV would consume ~16 GB on its own.

### `vendor_patches/dflash/` — DFlash C++ port fixes
Patches to the DFlash speculative decoding C++ implementation, specifically tuned for Qwen3.5-35B-A3B:
- Correct logit extraction from the last **accepted** position (not last drafted position)
- `--dflash-draft` properly wired through to `dflash_init`
- `--moe-topk` passthrough for baseline parity
- `--max-cycles` flag for cheap one-cycle acceptance probes
- GDN/recurrent architecture rollback via committed-prefix recompute

### `evodistill/` — Evolutionary LoRA distillation
Evolution-strategy optimiser that searches the LoRA adapter space for inference-speed improvements without quality regression. Uses the live inference binary as its fitness function — no separate training infrastructure needed.

### `scripts/autoresearch_loop.py` — Autonomous overnight research
Karpathy `autoresearch`-style loop adapted for inference optimisation: proposes flag/config changes, benchmarks with `llama-bench`, compares against a live control, keeps or discards, loops indefinitely. Results tracked in `results/autoresearch/ledger.tsv`.

---

## Repository layout

```
ANE-TurboStream/
├── docs/
│   ├── foundation/          # Master plan and immutable strategic documents
│   └── plans/               # Dated executable implementation plans
├── scripts/
│   ├── bench_protocol.sh    # Canonical reproducible benchmark runner
│   ├── autoresearch_loop.py # Autonomous overnight research loop
│   ├── dflash_benchmark.sh  # DFlash-specific benchmarks
│   └── ...                  # Draft conversion, build helpers
├── turbo_metal_port/        # Metal shaders for TurboQuant KV compression
├── vendor_patches/
│   └── dflash/              # C++ patches for DFlash speculative decoding
├── evodistill/              # Evolutionary LoRA distillation pipeline
├── dflash_mlx/              # MLX-based DFlash reference port
├── war_room/                # Active integration workspace and sprint docs
├── results/                 # Benchmark summaries and experiment ledger
├── PROGRESS.md              # Live progress tracker
└── RUNNING_SUMMARY.md       # Human-readable session narrative
```

Vendor dependencies (not committed — clone separately):

| Dependency | Source | Purpose |
|---|---|---|
| `vendor/anemll-flash-llama.cpp` | [anemll/anemll-flash-llama.cpp](https://github.com/anemll/anemll-flash-llama.cpp) | llama.cpp fork with Flash-MoE and DFlash |
| `vendor/llama-cpp-turboquant` | TurboQuant fork | `--cache-type-k turbo3` KV compression |

---

## Getting started

### Prerequisites

- Apple Silicon Mac (M1 or later), macOS 15+
- Xcode Command Line Tools (`xcode-select --install`)
- Model: `Qwen3.5-35B-A3B-UD-IQ2_M.gguf` (~10.6 GB) from HuggingFace

### Build the inference binary

```bash
git clone https://github.com/anemll/anemll-flash-llama.cpp vendor/anemll-flash-llama.cpp
cmake -B build-nsg-opt vendor/anemll-flash-llama.cpp \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_FLASH_MOE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-nsg-opt --target llama-bench llama-cli -j$(sysctl -n hw.logicalcpu)
```

### Run the baseline benchmark

```bash
MODEL=/path/to/Qwen3.5-35B-A3B-UD-IQ2_M.gguf
bash scripts/bench_protocol.sh "$MODEL"
```

Expected: two `llama-bench` tables (pp512, tg128) and a comparison summary.

### Run the AutoResearch loop overnight

```bash
MODEL=/path/to/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
LLAMA_BENCH=build-nsg-opt/bin/llama-bench \
python3 scripts/autoresearch_loop.py
```

Appends to `results/autoresearch/ledger.tsv`. Safe to interrupt and resume.

---

## Roadmap

| Track | Target | Status |
|---|---|---|
| Track 3 — DFlash AR (C++) | ≥60% acceptance, ≥38 t/s | Next |
| Track 11A — MPSGraphExecutable | +10–30% on LM-head | Week 1 |
| Track 2 — CoreML ANE dispatch | ANE > 0% measured | Week 2 |
| Track 7A — CPU+GPU+ANE concurrent dispatch | ≥40 t/s | Week 3 |
| Track 10 — ANE Long-Context Engine | 128K context on 16 GB | Month 1 |

---

## Methodology

Every result in this repo follows the same rules:

1. **Local benchmark truth beats theory.** If it doesn't measure faster, it isn't faster.
2. **llama.cpp is the foundation.** No framework switches without a benchmarked reason.
3. **Every optimisation ships with:** exact command, exact benchmark artifact, and explicit accept/reject criteria.
4. **No ANE acceleration claims without measured proof** — `mactop` showing ANE > 0%, `aneDCSBytes` channel from IOReport.
5. **Reproducibility gate:** `scripts/bench_protocol.sh` must produce results within ±2% before any result is committed.

---

## Papers & references

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — ICLR 2026
- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) — 2026
- [LLM in a Flash](https://arxiv.org/abs/2312.11514) — Apple Research
- [PolarQuant](https://arxiv.org/abs/2502.02617) — AISTATS 2026

---

## License

Research code — MIT. Vendor dependencies retain their original licences.
