# Phase 1 Notes

Date: 2026-04-04

## Fresh extraction attempt

Attempted command:

`./.venv-tools/bin/python vendor/anemll-flash-llama.cpp/tools/flashmoe-sidecar/flashmoe_sidecar.py extract --model /Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf --out-dir /Users/manuelmonteiro/Desktop/ANE project/results/sidecar/qwen35 --force`

Result:
- failed with `OSError: [Errno 28] No space left on device`

Observed machine state during attempt:
- data volume available space was approximately `1.0 GiB`

## Interim fallback

Because the fresh clean-room extraction could not finish due disk pressure, I switched to validating the previously extracted Flash-MoE-compatible sidecar located at:

`/Users/manuelmonteiro/Desktop/ANE project_archive_20260404_134904/sidecar_output/qwen35`

This archived sidecar appears to match the expected Flash-MoE GGUF format:
- `schema_version: 1`
- `sidecar_kind: flashmoe_gguf`
- `layout: layer_major_whole_tensor`
- source model path points to `/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf`

## Rule

Use the archived sidecar only as an interim read-only artifact until either:
1. disk space is freed for a fresh extraction, or
2. verification proves the archived sidecar is valid and reproducible enough for Phase 1 measurement work.
