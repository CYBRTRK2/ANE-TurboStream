#!/usr/bin/env python3
"""Build draft GGUF by copying tokenizer metadata from target, then injecting draft tensors."""

import os
import shutil
import numpy as np
import gguf

TARGET = os.path.expanduser("~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf")
DRAFT_SRC = os.path.expanduser("~/models/Qwen3.5-35B-A3B-Draft-f16.gguf")
OUTPUT = os.path.expanduser("~/models/Qwen3.5-35B-A3B-Draft-f16-fixed.gguf")

# Step 1: Copy target as base (it has perfect tokenizer)
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)
shutil.copy2(TARGET, OUTPUT)
print(f"Copied target to {OUTPUT}")

# Step 2: Read draft tensors from existing draft GGUF
draft_reader = gguf.GGUFReader(DRAFT_SRC)
draft_tensors = {}
for t in draft_reader.tensors:
    name = t.name
    raw_data = np.array(t.data)
    # GGUF reader returns data as raw bytes; reshape
    shape = t.shape
    # Determine dtype from tensor_type
    if t.tensor_type == 1:  # GGML_TYPE_F32
        arr = raw_data.view(np.float32).reshape(shape)
    elif t.tensor_type == 33:  # GGML_TYPE_F16
        arr = raw_data.view(np.float16).reshape(shape)
    else:
        # Read as bytes and try to cast
        arr = np.frombuffer(raw_data.tobytes(), dtype=np.float16).reshape(shape)
    draft_tensors[name] = arr.copy()
    print(f"  Read draft tensor: {name} shape={shape} dtype={arr.dtype}")

print(f"Loaded {len(draft_tensors)} draft tensors")
