# Autoresearch Adaptation for ANE-TurboStream

We are explicitly adapting `papers:references:repos/autoresearch-macos-master` into this project.

## Why it matters

Karpathy's autoresearch pattern is the right control loop for both:
1. inference optimization
2. future forward-only self-improvement

The key principle is simple:
- try one bounded idea
- measure it
- keep only real gains
- discard regressions
- continue autonomously

## Loop A: Inference optimization

Champion state:
- current best clean-room runtime configuration
- exact benchmark command
- exact quality/perplexity status

Per experiment:
1. make one bounded runtime/config change
2. run benchmark script
3. run quality gate
4. log results as keep/discard/crash
5. advance champion only if metrics improve without unacceptable regressions

Required metrics:
- decode tok/s
- prefill tok/s
- quality gate pass/fail
- perplexity delta when relevant
- memory / OOM status
- notes on variance

## Loop B: Forward-only improvement

Champion state:
- current best adapter / merged checkpoint
- eval score
- quality gate status

Per experiment:
1. generate corpus or select training batch
2. run forward-only update (e.g. LoRA + ES / SSD-style self-distillation)
3. evaluate on fixed held-out set
4. log keep/discard/crash
5. advance champion only on measured improvement

Required metrics:
- val loss / pass@1 / task metric
- quality gate pass/fail
- training/eval wall-clock
- memory usage

## Project rule

No optimization idea is considered real until it survives the autoresearch-style loop and becomes the new champion via measured local results.
