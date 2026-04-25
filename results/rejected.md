
## 2026-04-25 Track 1 — Free wins on existing binary

### llama-lookahead
- Error: split_equal with coupled sequences (GatedDeltaNet/M-RoPE incompatibility)
- Verdict: NOT FIXABLE without rewriting graph split logic. CLOSED.

### llama-lookup
- Result: 5.1 0x0p+0cceptance, 18.08 t/s (slower than 26.02 baseline)
- Verdict: Regresses speed. CLOSED.

### --moe-mode resident
- Error: Requires sidecar files (not present)
- Verdict: CLOSED until sidecar generation viable.
