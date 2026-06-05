# Phase 0 Summary

Date: 2026-04-04
Model: `/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf`
Runtime base: `vendor/anemll-flash-llama.cpp`
Build: `da3b409e` (`b8448-da3b409e` in runtime output)
Mode: `--moe-mode stock`

## Benchmark command

`/Users/manuelmonteiro/Desktop/ANE project/vendor/anemll-flash-llama.cpp/build/bin/llama-bench -m /Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf --moe-mode stock -ngl 99 -p 64 -n 32 -r 1 --no-warmup -o md`

## First clean-room measurements

- Prompt processing (`pp64`): `59.82 t/s`
- Token generation (`tg32`): `22.65 t/s`

Source artifact:
- `results/p0_stock_baseline.md`

## Smoke generation

Prompt:
- `The capital of Portugal is`

Command notes:
- `--reasoning off`
- `-st`
- `--temp 0`
- `-ngl 99`

Observed output:
- contains `Lisbon`

Source artifact:
- `results/p0_stock_smoke.txt`

## Notes

- The initial smoke command hung because `llama-cli` in this fork does not support `--no-conversation`; the script was corrected to use `-st` and `--reasoning off`.
- This is the first clean-room baseline. Repeated runs are still required before treating this as a fully defended long-term anchor.
