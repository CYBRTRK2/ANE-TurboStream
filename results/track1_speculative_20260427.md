# Track 1 Speculative Alternatives — 2026-04-27

Model: `/Users/manuelmonteiro/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf`  
Build: `build-nsg-opt`, build 8448  
Flags unless noted: `-ngl 99 -t 4 --moe-mode stock --moe-topk 4`, `GGML_COREML_ENABLE=0`

## Control

`dflash-cli --baseline -c 512 -n 64 -p "The capital of Portugal is"`:

- Post-prefill decode: `21.30 tok/s`
- End-to-end: `19.28 tok/s`
- This is a live-session control only. Official Apr 25 topk=4 baseline remains `26.02 +/- 0.25 tok/s`.

## llama-lookup

### Repeated-context positive probe

Prompt:

`The capital of Portugal is Lisbon. The capital of Portugal is Lisbon. The capital of Portugal is`

Command used `llama-lookup --draft 16 --draft-min 0 -lcd results/lookup_repeated_adaptive_20260427.ngram -n 64`.

Result:

- Decode: `79.807 tok/s`
- Accepted: `64/64`
- Recomputes: `0`
- Adaptive status: `active`

Conclusion: lookup decoding works and can be a large win when the prompt/context contains reusable n-grams.

### General-prompt correctness probe

Prompt:

`Explain why the sky appears blue during the day in one concise paragraph.`

Initial unpatched behavior was invalid: `llama_decode` emitted M-RoPE position errors but the example continued and reported an inflated speed.

Fixes applied:

- `examples/lookup/lookup.cpp` now recomputes the committed prefix when `llama_memory_seq_rm` cannot crop recurrent/GDN state.
- Verification decode failures now return non-zero instead of silently continuing.
- Added adaptive n-gram disable after repeated recomputes with low acceptance.

After the fixes:

- Decode: `12.752 tok/s`
- Accepted: `8/64`
- Recomputes: `4`
- Adaptive status: `disabled`

Conclusion: lookup is not a daily-driver accelerator for normal prompts on this recurrent/M-RoPE model. It is useful only for repeated/cacheable text unless a cheaper recurrent rollback path is implemented.

## llama-lookahead

Fixes applied:

- `examples/lookahead/lookahead.cpp` now exposes `LLAMA_LOOKAHEAD_W`, `LLAMA_LOOKAHEAD_N`, and `LLAMA_LOOKAHEAD_G`.
- Prompt decode failures now return non-zero.
- Shared memory splitters now allow non-sequential equal splitting for coupled sequence batches.

Results:

- Default W=15/G=15 allocates 31 recurrent sequences and OOMs on the current M4 Air memory envelope.
- Small probe `LLAMA_LOOKAHEAD_W=1 LLAMA_LOOKAHEAD_G=1 LLAMA_LOOKAHEAD_N=2 -c 256 -n 16` gets past the original coupled-sequence split error, but still fails later with stale M-RoPE positions for auxiliary sequences.

Conclusion: lookahead remains blocked. It needs a recurrent/M-RoPE-safe auxiliary sequence cleanup or recompute strategy before it can be benchmarked honestly.

