# DFlash Smoke — 2026-04-27

## Build

```bash
cmake --build build-nsg-opt --target dflash-cli -j 4
```

## Findings

- Block fallback no longer duplicates the staged first token.
- Next-cycle logits now come from `accepted_length`, not the end of the whole verified block.
- Partial rollback attempts `llama_memory_seq_rm`; for Qwen3.5 GDN/recurrent state this often fails, so the safe fallback is to clear and recompute the committed prefix.
- `--max-cycles N` was added for cheap one-cycle acceptance probes.

## Probe

```bash
./build-nsg-opt/bin/dflash-cli \
  -m /Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --dflash --moe-topk 4 \
  -p 'The capital of Portugal is' \
  -n 16 -c 1024 --block-size 16 --max-cycles 1
```

Result: placeholder copy-token drafting accepted `0/15`, generated `Lisbon`, and incurred no replay/recompute cost because `--max-cycles 1` stopped after one verify cycle.

## Draft File Probe

`/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-Draft-f16-tokenizerfix.gguf` loads tokenizer metadata correctly, but llama.cpp rejects it as a normal Qwen3 model:

```text
missing tensor 'token_embd.weight'
```

This means the current DFlash draft cannot be used through normal llama.cpp autoregressive model loading. The remaining viable paths are a custom DFlash draft runtime or adding compatible embedding/head tensors with careful memory accounting.
