# Phase 4 Revisit: ANE Private API Results

**Date**: 2026-04-10
**Hardware**: M4 MacBook Air 16GB (Apple T8132)
**Status**: BREAKTHROUGH — ANE fully accessible via private API

## Executive Summary

Previous Phase 4 concluded "ANE is counterproductive for decode" based on micro-benchmarks of single-vector GEMV (batch=1). This revisit tested the CORRECT use cases: batched GEMM and convolution operations. **The ANE is fully functional and 3.5-5x faster than CPU for convolution workloads.**

More importantly, we achieved **direct ANE access via the private API** (`_ANEInMemoryModel`, `_ANERequest`, `_ANEIOSurfaceObject`), bypassing CoreML's overhead and getting 11% faster inference.

## ANE Hardware (confirmed via `_ANEDeviceInfo`)

| Property | Value |
|---|---|
| hasANE | YES |
| numANEs | 1 |
| numANECores | **16** |
| aneSubType | 0x1FA6122D8 |
| aneArchitectureType | 0xBC9C5D8551A370A6 |
| aneSubTypeVariant | 0x1FA612358 |
| IOKit driver | H11ANEIn (AppleH16ANEInterface) |
| HAL | AppleT8132ANEHAL |
| Load balancer | H1xANELoadBalancer (6 DirectPathClients active) |
| Compiler service | ANECompilerServi (pid 76512) |
| productName | macOS |
| buildVersion | 25E246 |

## Private API Pipeline (Path B) — WORKS END-TO-END

```
_AENEInMemoryModelDescriptor  →  _ANEInMemoryModel  →  compile  →  load  →  evaluate
       ↓                              ↓                    ↓          ↓          ↓
   modelWithMILText:         inMemoryModelWith     compileWith   loadWith   evaluateWith
   (NSData MIL binary)      Descriptor:             QoS:        QoS:       QoS:
                              (descriptor)
```

### Key Steps

1. Load CoreML-compiled `model.mil` (binary MIL text) as `NSData`
2. Create `_ANEInMemoryModelDescriptor` via `modelWithMILText:weights:optionsPlist:`
3. Create `_ANEInMemoryModel` via `inMemoryModelWithDescriptor:`
4. Write `model.mil` and `weights/weight.bin` to temp dir matching `hexStringIdentifier`
5. Compile: `compileWithQoS:options:error:` → SUCCESS
6. Load: `loadWithQoS:options:error:` → SUCCESS
7. Create `IOSurface` for I/O, wrap in `_ANEIOSurfaceObject`
8. Create `_ANERequest` with inputs/outputs
9. Evaluate: `evaluateWithQoS:options:request:error:` → SUCCESS

### What Doesn't Work

- **Freeform MIL text compilation**: `ANECCompile() FAILED` with `InvalidMILProgram`
- The ANE compiler requires MIL in CoreML's compiled binary format (protobuf-based)
- Weights must be in `BLOBFILE` format as CoreML generates
- `_ANEClient.compileModel:` and `_ANEClient.loadModel:` expect `_ANEInMemoryModelDescriptor` objects (not file paths)

## Benchmarks

### Deep ConvNet (5-layer + FC, ~8.5M weights)

| Compute Unit | Latency | Throughput |
|---|---|---|
| **Private API (_ANEInMemoryModel)** | **0.579 ms** | **1,850/sec** |
| CoreML ALL (ANE preferred) | 0.652 ms | ~1,500/sec |
| CoreML CPU+NE | 0.67 ms | ~1,500/sec |
| CoreML CPU+GPU | 4.08 ms | ~245/sec |
| CoreML CPU_ONLY | 3.04 ms | ~330/sec |

### Sustained Burn (10 seconds)

| Compute Unit | Inferences | Rate |
|---|---|---|
| ANE (ALL) | 11,732 | 1,173/sec |
| CPU_ONLY | 3,323 | 332/sec |
| **ANE speedup** | | **3.5x** |

### Simple ConvNet (4-layer, ~1.7M weights)

| Compute Unit | Latency |
|---|---|
| ALL (ANE preferred) | 2.317 ms |
| CPU+NE | 2.037 ms |
| CPU_ONLY | 7.511 ms |

### MLP SwiGLU (2048→1408→2048)

| Batch | ALL (ANE) | CPU+GPU | CPU_ONLY |
|---|---|---|---|
| 1 | 0.22 ms | 0.17 ms | 0.20 ms |
| 4 | 0.31 ms | 0.24 ms | 0.38 ms |
| 32 | 1.89 ms | 1.18 ms | 2.60 ms |

### LM-head projection (2048→248320)

| Batch | ALL (ANE) | CPU_ONLY |
|---|---|---|
| 1 | ~1.2 ms | ~1.1 ms |
| 4 | ~1.5 ms | ~1.5 ms |

**Key insight**: ANE is faster for conv operations but NOT for GEMM/MLP. The ANE's architecture is optimized for spatial convolutions, not dense matmuls.

## Private API Classes Discovered

### _ANEClient (46 instance methods)
Key methods:
- `compileModel:options:qos:error:` — compile a model descriptor
- `loadModel:options:qos:error:` — load compiled model
- `evaluateWithModel:options:request:error:` — run inference
- `evaluateRealTimeWithModel:options:request:error:` — real-time inference
- `loadRealTimeModel:options:qos:error:` — load for real-time
- `mapIOSurfacesWithModel:request:cacheInference:error:` — zero-copy I/O
- `beginRealTimeTask` / `endRealTimeTask` — real-time priority
- `compileModel:options:qos:error:` — expects descriptor object
- `doEvaluateDirectWithModel:options:request:qos:error:` — direct evaluation
- `doPrepareChainingWithModel:options:chainingReq:error:` — model chaining
- `compiledModelExistsFor:` / `compiledModelExistsMatchingHash:` — cache lookup

### _ANERequest
- `requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:` (multiple variants)

### _ANEIOSurfaceObject
- `objectWithIOSurface:` — wraps IOSurface for ANE I/O

### _ANEDeviceInfo
- Class methods: `hasANE`, `numANEs`, `numANECores`, `aneSubType`, `aneArchitectureType`, `aneSubTypeVariant`, `aneBoardType`, `productName`, `buildVersion`, `isVirtualMachine`, `precompiledModelChecksDisabled`

## Implications for ANE-TurboStream

1. **ANE is real and accessible** — the Phase 4 conclusion was wrong for the right workloads
2. **Conv layers can use ANE** — attention QKV projections, conv-based operators
3. **GEMM/MLP is faster on CPU** — ANE adds overhead for dense matmuls
4. **Private API gives ~11% speedup** over CoreML's predict() for conv workloads
5. **IOSurface zero-copy** means we can chain ANE operations without memcpy
6. **Model chaining** (`doPrepareChainingWithModel:`) could enable multi-op pipelines
7. **Real-time API** (`evaluateRealTimeWithModel:`) could reduce latency for interactive use

### Hybrid Strategy

For the Qwen3.5-35B-A3B model:
- **Attention (QKV/O projections)**: Run as conv-like ops on ANE via private API
- **MLP (gate/up/down)**: Keep on CPU (GEMM is faster there)
- **LM-head**: Keep on CPU (too large for ANE, no speedup)
- **IOSurface pipeline**: Zero-copy data movement between ANE and CPU

## Files

- `/tmp/ane_pathb_v7.m` — Working private API POC (compile + load + evaluate)
- `/tmp/ane_compiled_model/` — CoreML-compiled model.mil + weights
- `/tmp/ane_deep_conv.mlpackage` — Deep ConvNet CoreML model
- `/tmp/ane_stress_conv.mlpackage` — Simple ConvNet CoreML model
- Original POC: `pocs/ane_direct.m` in apple-silicon-internals repo