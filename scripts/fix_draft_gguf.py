#!/usr/bin/env python3
"""Add missing token_embd and output weights to the DFlash draft GGUF.

The DFlash draft model shares token embeddings and LM head with the target model.
For llama.cpp speculative decoding, the draft needs these weights included.

This script reads token_embd.weight and output.weight from the target GGUF,
converts them to f16, and adds them to the draft GGUF by reconstructing it.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Desktop/ANE project/vendor/anemll-flash-llama.cpp"))
from gguf import GGUFReader, GGUFWriter, GGUFValueType

DRAFT_GGUF = os.path.expanduser("~/models/Qwen3.5-35B-A3B-Draft-Q4KM.gguf")
TARGET_GGUF = os.path.expanduser("~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf")

print("Reading target model for embedding/LM head weights...")
reader = GGUFReader(TARGET_GGUF)

# Find the token_embd and output tensors
token_embd = None
output_weight = None

for tensor in reader.tensors:
    if tensor.name == "token_embd.weight":
        token_embd = tensor
        print(f"  Found token_embd.weight: shape={tensor.shape}, type={tensor.data_type}")
    elif tensor.name == "output.weight":
        output_weight = tensor
        print(f"  Found output.weight: shape={tensor.shape}, type={tensor.data_type}")

if token_embd is None:
    # Try alternate names
    for tensor in reader.tensors:
        if "embed" in tensor.name.lower():
            print(f"  Alt: {tensor.name}: shape={tensor.shape}")
        if "output" in tensor.name.lower() or "lm_head" in tensor.name.lower():
            print(f"  Alt: {tensor.name}: shape={tensor.shape}")

# Read draft GGUF
print("\nReading draft GGUF...")
draft_reader = GGUFReader(DRAFT_GGUF)

print(f"  Draft has {len(draft_reader.tensors)} tensors")
print(f"  Draft has {len(draft_reader.fields)} fields")

# Check what tensors the draft has
draft_tensor_names = [t.name for t in draft_reader.tensors]
print(f"  Draft tensors: {sorted(draft_tensor_names)[:5]}...{sorted(draft_tensor_names)[-5:]}")

# Check if embed/lm_head are already there
for name in ["token_embd.weight", "output.weight"]:
    if name in draft_tensor_names:
        print(f"  {name} already in draft GGUF!")
    else:
        print(f"  {name} MISSING from draft GGUF - needs to be added")