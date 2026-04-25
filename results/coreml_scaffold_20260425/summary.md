# Track 2 Scaffold — CoreML+Metal Build Verification

## Build
- Tree: vendor/anemll-flash-llama.cpp/build-coreml-metal/
- Config: GGML_COREML=ON, GGML_METAL=ON, Release, static libs
- Status: llama-cli + llama-bench built successfully
- CoreML framework linked (otool confirms)

## Benchmark (topk=4, ngl=99, pp512, tg128)
- pp512: 334.40 +/- 2.82 t/s
- tg128: 23.96 +/- 0.20 t/s
- Backend: MTL,BLAS (CoreML registered but dormant)

## Observation
- CoreML backend is linked but no ops match yet (scaffold returns false)
- Baseline tg128 = 23.96 t/s (consistent with stock after accounting for build)

## Next
Implement MIL compilation in ggml-coreml-impl.mm for shared-expert FFN.
Gate: mactop ANE % > 0 during decode.
