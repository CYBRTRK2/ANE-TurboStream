#!/usr/bin/env python3
"""Convert DFlash draft model to GGUF using MLX for bf16 handling.

MLX can read bf16 safetensors natively. We convert to float32 then write GGUF.
"""
import os
import sys
import subprocess
import numpy as np

# Step 1: Use MLX to convert bf16 safetensors to f16 safetensors (or numpy)
# Then use our GGUF writer

sys.path.insert(0, os.path.expanduser("~/Desktop/ANE project/vendor/anemll-flash-llama.cpp"))
from gguf import GGUFWriter, GGUFValueType

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash/"
    "snapshots/a6ab3a277f856d91c43f28711611e7929073d56d"
)
OUTPUT_PATH = os.path.expanduser("~/models/Qwen3.5-35B-A3B-Draft-f16.gguf")

def map_tensor_name(name):
    """Map DFlash tensor names to Qwen3 GGUF format."""
    if name in ("fc.weight", "hidden_norm.weight"):
        return None
    if name.startswith("layers."):
        parts = name.split(".", 2)
        layer = parts[1]
        rest = parts[2]
        mapping = {
            "input_layernorm.weight": f"blk.{layer}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{layer}.ffn_norm.weight",
            "self_attn.q_proj.weight": f"blk.{layer}.attn_q.weight",
            "self_attn.k_proj.weight": f"blk.{layer}.attn_k.weight",
            "self_attn.v_proj.weight": f"blk.{layer}.attn_v.weight",
            "self_attn.o_proj.weight": f"blk.{layer}.attn_output.weight",
            "self_attn.q_norm.weight": f"blk.{layer}.attn_q_norm.weight",
            "self_attn.k_norm.weight": f"blk.{layer}.attn_k_norm.weight",
            "mlp.gate_proj.weight": f"blk.{layer}.ffn_gate.weight",
            "mlp.up_proj.weight": f"blk.{layer}.ffn_up.weight",
            "mlp.down_proj.weight": f"blk.{layer}.ffn_down.weight",
        }
        mapped = mapping.get(rest)
        if mapped is None:
            print(f"  WARNING: Unknown tensor {name}, skipping")
        return mapped
    if name == "norm.weight":
        return "output_norm.weight"
    print(f"  WARNING: Unknown tensor {name}, skipping")
    return None


def main():
    print("Loading model with MLX...")
    import mlx.core as mx
    import mlx.nn as nn
    
    # Load weights via MLX - it handles bf16 natively
    from mlx.utils import tree_flatten
    
    # Read the safetensors file directly using mlx
    weights = mx.load(os.path.join(MODEL_DIR, "model.safetensors"))
    
    print(f"Loaded {len(weights)} tensors")
    print(f"Creating GGUF: {OUTPUT_PATH}")
    
    writer = GGUFWriter(OUTPUT_PATH, "qwen3", use_temp_file=True)
    
    # Model metadata
    n_layers = 8
    n_heads = 32
    n_kv_heads = 4
    hidden_size = 2048
    intermediate_size = 6144
    head_dim = 128
    vocab_size = 248320
    rms_norm_eps = 1e-6
    rope_theta = 10000000.0

    writer.add_name("Qwen3.5-35B-A3B-DFlash-Draft")
    writer.add_context_length(262144)
    writer.add_embedding_length(hidden_size)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_rope_freq_base(rope_theta)
    writer.add_vocab_size(vocab_size)
    writer.add_uint32("n_head_dim", head_dim)

    # Add tokenizer from HuggingFace tokenizer files
    # The draft model shares the same tokenizer as target Qwen3.5-35B-A3B
    print("Adding tokenizer from HuggingFace...")
    import json
    
    tokenizer_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/"
        "snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
    )
    
    # Load tokenizer.json
    with open(os.path.join(tokenizer_dir, "tokenizer.json")) as f:
        tok_json = json.load(f)
    
    model_type = tok_json.get("model", {}).get("type", "gpt2")
    # Use 'gpt2' not 'BPE' - llama.cpp expects the ggml tokenizer model name
    writer.add_tokenizer_model("gpt2")
    
    # Pre tokenizer - llama.cpp needs specific pre tokenizer names
    writer.add_tokenizer_pre("qwen35")
    
    # Tokens
    added_tokens = tok_json.get("added_tokens", [])
    vocab = tok_json.get("model", {}).get("vocab", {})
    
    # Build sorted token list
    tokens_by_id = {}
    for token, idx in vocab.items():
        tokens_by_id[idx] = token
    for at in added_tokens:
        tokens_by_id[at["id"]] = at["content"]
    
    max_id = max(tokens_by_id.keys()) if tokens_by_id else 0
    all_tokens = []
    all_types = []
    for i in range(max_id + 1):
        if i in tokens_by_id:
            all_tokens.append(tokens_by_id[i])
            # Check if it's a special/added token
            is_special = any(at["id"] == i and at.get("special", False) for at in added_tokens)
            all_types.append(3 if is_special else 1)  # 3=SPECIAL, 1=NORMAL
        else:
            all_tokens.append(f"<UNUSED_{i}>")
            all_types.append(0)
    
    writer.add_token_list(all_tokens)
    writer.add_token_types(all_types)
    
    # Merges
    merges = tok_json.get("model", {}).get("merges", [])
    writer.add_token_merges(merges)
    
    # EOS token
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json")) as f:
        tok_cfg = json.load(f)
    
    eos_id = tok_cfg.get("eos_token_id", 248046)
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    writer.add_uint32("tokenizer.ggml.eos_token_id", eos_id)
    
    pad_id = tok_cfg.get("pad_token_id", 248055)
    if pad_id is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", pad_id)
    
    # Chat template
    chat_template = tok_cfg.get("chat_template", "")
    if chat_template:
        if isinstance(chat_template, list):
            chat_template = chat_template[0].get("template", "")
        writer.add_string("tokenizer.chat_template", chat_template)
    
    print(f"  Added {len(all_tokens)} tokens, {len(merges)} merges, eos={eos_id}")

    total_size = 0
    skipped = 0
    added = 0

    for name in sorted(weights.keys()):
        gguf_name = map_tensor_name(name)
        if gguf_name is None:
            skipped += 1
            print(f"  SKIP: {name}")
            continue

        # MLX tensor -> float16 numpy via MLX float32 conversion
        arr = mx.array(weights[name], dtype=mx.float32)
        data = np.array(arr, dtype=np.float16)
        data = np.ascontiguousarray(data)
        total_size += data.nbytes

        print(f"  ADD: {name} -> {gguf_name} {data.shape} {data.nbytes/1e6:.1f}MB")
        writer.add_tensor(gguf_name, data)
        added += 1

    print(f"\nAdded: {added} tensors, Skipped: {skipped} tensors")
    print(f"Total size: {total_size/1e9:.2f} GB")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    out_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nDone! GGUF written to: {OUTPUT_PATH}")
    print(f"File size: {out_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()