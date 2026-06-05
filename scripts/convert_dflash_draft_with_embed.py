#!/usr/bin/env python3
"""
Rebuild the draft GGUF with token_embd and output weights from the target model.
DFlash shares these with the target, but llama.cpp requires them in the draft GGUF.
"""
import sys
import os
import numpy as np
import mlx.core as mx
import json

sys.path.insert(0, os.path.expanduser("~/Desktop/ANE project/vendor/anemll-flash-llama.cpp"))
from gguf import GGUFWriter, GGUFReader

MODEL_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash/"
    "snapshots/a6ab3a277f856d91c43f28711611e7929073d56d"
)
TOKENIZER_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/"
    "snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
)
TARGET_GGUF = os.path.expanduser("~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf")
OUTPUT_PATH = os.path.expanduser("~/models/Qwen3.5-35B-A3B-Draft-f16.gguf")

def map_tensor_name(name):
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
        return mapping.get(rest)
    if name == "norm.weight":
        return "output_norm.weight"
    return None


def main():
    print("Loading draft model weights...")
    weights = mx.load(os.path.join(MODEL_DIR, "model.safetensors"))

    print(f"Creating GGUF: {OUTPUT_PATH}")
    writer = GGUFWriter(OUTPUT_PATH, "qwen3", use_temp_file=True)

    # Model metadata
    writer.add_name("Qwen3.5-35B-A3B-DFlash-Draft")
    writer.add_context_length(262144)
    writer.add_embedding_length(2048)
    writer.add_block_count(8)
    writer.add_feed_forward_length(6144)
    writer.add_head_count(32)
    writer.add_head_count_kv(4)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(10000000.0)
    writer.add_vocab_size(248320)
    writer.add_uint32("n_head_dim", 128)

    # Tokenizer
    print("Adding tokenizer...")
    with open(os.path.join(TOKENIZER_DIR, "tokenizer.json")) as f:
        tok_json = json.load(f)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35")
    
    added_tokens = tok_json.get("added_tokens", [])
    vocab = tok_json.get("model", {}).get("vocab", {})
    tokens_by_id = {}
    for token, idx in vocab.items():
        tokens_by_id[idx] = token
    for at in added_tokens:
        tokens_by_id[at["id"]] = at["content"]
    max_id = max(tokens_by_id.keys())
    all_tokens = []
    all_types = []
    for i in range(max_id + 1):
        if i in tokens_by_id:
            all_tokens.append(tokens_by_id[i])
            is_special = any(at["id"] == i and at.get("special", False) for at in added_tokens)
            all_types.append(3 if is_special else 1)
        else:
            all_tokens.append(f"<UNUSED_{i}>")
            all_types.append(0)
    writer.add_token_list(all_tokens)
    writer.add_token_types(all_types)
    writer.add_token_merges(tok_json.get("model", {}).get("merges", []))
    
    with open(os.path.join(TOKENIZER_DIR, "tokenizer_config.json")) as f:
        tok_cfg = json.load(f)
    eos_id = tok_cfg.get("eos_token_id", 248046)
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    writer.add_uint32("tokenizer.ggml.eos_token_id", eos_id)
    pad_id = tok_cfg.get("pad_token_id", 248055)
    if pad_id is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", pad_id)
    chat_template = tok_cfg.get("chat_template", "")
    if chat_template:
        if isinstance(chat_template, list):
            chat_template = chat_template[0].get("template", "")
        writer.add_string("tokenizer.chat_template", chat_template)
    print(f"  {len(all_tokens)} tokens, eos={eos_id}")

    # Draft model weights
    print("Adding draft weights...")
    total_size = 0
    for name in sorted(weights.keys()):
        gguf_name = map_tensor_name(name)
        if gguf_name is None:
            continue
        arr = mx.array(weights[name], dtype=mx.float32)
        data = np.ascontiguousarray(np.array(arr, dtype=np.float16))
        total_size += data.nbytes
        writer.add_tensor(gguf_name, data)

    # Add token_embd and output from target model (tied embeddings)
    # Read from target GGUF
    print("Adding shared embedding/LM head from target model...")
    target_reader = GGUFReader(TARGET_GGUF)
    
    for tensor in target_reader.tensors:
        if tensor.name == "output.weight":
            # This is the LM head / tied embedding: [2048, 248320]
            print(f"  Adding {tensor.name}: shape={tensor.shape}")
            # The tensor is quantized in the target (iq2_m), we need f16 for the draft
            # Actually, for speculative decoding, we just need the draft to produce logits
            # We can use a random/zero embedding since draft tokens are verified by target
            # But for better acceptance, we should use the real embedding.
            # Since the target GGUF is IQ2_M quantized, we need to dequantize first.
            # This is complex. For now, use a zero/random embedding.
            # Better approach: use the output.weight from the target model directly.
            # GGUFReader gives us the dequantized data.
            data = np.array(tensor)
            print(f"    Dequantized shape: {data.shape}, dtype: {data.dtype}")
            data_f16 = np.ascontiguousarray(data.astype(np.float16))
            total_size += data_f16.nbytes
            # Use tied embeddings: token_embd = output
            writer.add_tensor("token_embd.weight", data_f16)
            writer.add_tensor("output.weight", data_f16)
            print(f"    Added as f16: {data_f16.nbytes/1e6:.1f}MB each")
            break

    print(f"\nTotal size: {total_size/1e9:.2f} GB")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    out_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nDone! GGUF written to: {OUTPUT_PATH}")
    print(f"File size: {out_size/1e9:.2f} GB")

if __name__ == "__main__":
    main()