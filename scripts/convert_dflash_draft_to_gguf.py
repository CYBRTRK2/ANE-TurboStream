#!/usr/bin/env python3
"""Convert DFlash draft model (safetensors) to GGUF for llama.cpp speculative decoding.

The DFlash draft model is a standard Qwen3-style transformer with 8 layers.
We map it to llama.cpp's Qwen3 architecture so it can be used as --model-draft.
The fc.weight and hidden_norm.weight (DFlash-specific target hidden state projection)
are skipped - they're only needed for block-diffusion drafting, not AR drafting.
"""

import os
import sys
import numpy as np
from safetensors import safe_open

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-DFlash/"
    "snapshots/a6ab3a277f856d91c43f28711611e7929073d56d"
)
OUTPUT_PATH = os.path.expanduser(
    os.environ.get("DFLASH_DRAFT_OUTPUT", "~/models/Qwen3.5-35B-A3B-Draft-f16-loadable-metadata.gguf")
)

F16 = True  # Use float16
Q4KM = False  # Alternative: use Q4_K_M for smaller size
KEEP_DFLASH_CUSTOM = os.environ.get("DFLASH_KEEP_CUSTOM") == "1"

import gguf
from gguf import GGUFWriter, GGMLQuantizationType

def map_tensor_name(name):
    """Map DFlash tensor names to llama.cpp GGUF Qwen3 format, preserving FC."""
    # DFlash-specific tensors are useful for a future custom block-diffusion
    # loader, but standard llama.cpp model loading rejects unused tensors.
    if name == "fc.weight":
        return "dflash_fc.weight" if KEEP_DFLASH_CUSTOM else None
    if name == "hidden_norm.weight":
        return "dflash_norm.weight" if KEEP_DFLASH_CUSTOM else None

    # Layer tensors
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
            return None
        return mapped

    if name == "norm.weight":
        return "output_norm.weight"

    print(f"  WARNING: Unknown tensor {name}, skipping")
    return None


def main():
    print(f"Reading model from: {MODEL_PATH}")
    
    # Use MLX to load bf16 tensors (safetensors/pt fails on bf16)
    try:
        import mlx.core as mx
        weights = mx.load(os.path.join(MODEL_PATH, "model.safetensors"))
        print(f"Loaded {len(weights)} tensors via MLX")
        use_mlx = True
    except ImportError:
        from safetensors import safe_open
        sf = safe_open(os.path.join(MODEL_PATH, "model.safetensors"), framework="np")
        use_mlx = False
        print(f"Loaded via safetensors/numpy")

    # Model metadata for Qwen3 architecture in GGUF
    n_layers = 8
    n_heads = 32
    n_kv_heads = 4
    hidden_size = 2048
    intermediate_size = 6144
    head_dim = 128
    vocab_size = 248320
    rms_norm_eps = 1e-6
    rope_theta = 10000000.0

    print(f"Creating GGUF: {OUTPUT_PATH}")
    writer = GGUFWriter(OUTPUT_PATH, "qwen3", use_temp_file=True)

    # Write metadata
    writer.add_name("Qwen3.5-35B-A3B-DFlash-Draft")
    writer.add_context_length(262144)
    writer.add_embedding_length(hidden_size)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(rope_theta)
    writer.add_vocab_size(vocab_size)
    
    # Tokenizer metadata copied from target model
    # Required by llama.cpp to load the model (GPT-2 tokenizer backend)
    target_reader = gguf.GGUFReader(
        os.path.expanduser("~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf")
    )
    
    def _contents(key):
        return target_reader.fields[key].contents()

    tokens = _contents("tokenizer.ggml.tokens")
    token_types = _contents("tokenizer.ggml.token_type")
    merges = _contents("tokenizer.ggml.merges")
    if len(tokens) != len(token_types):
        raise ValueError(
            f"Tokenizer metadata mismatch: {len(tokens)} tokens vs {len(token_types)} token types"
        )

    writer.add_tokenizer_model(_contents("tokenizer.ggml.model"))
    writer.add_tokenizer_pre(_contents("tokenizer.ggml.pre"))
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_eos_token_id(int(_contents("tokenizer.ggml.eos_token_id")))
    writer.add_pad_token_id(int(_contents("tokenizer.ggml.padding_token_id")))

    if "tokenizer.ggml.bos_token_id" in target_reader.fields:
        writer.add_bos_token_id(int(_contents("tokenizer.ggml.bos_token_id")))
    if "tokenizer.ggml.unknown_token_id" in target_reader.fields:
        writer.add_unk_token_id(int(_contents("tokenizer.ggml.unknown_token_id")))

    if "tokenizer.chat_template" in target_reader.fields:
        writer.add_string("tokenizer.chat_template", _contents("tokenizer.chat_template"))

    # Process tensors
    total_size = 0
    skipped = 0
    added = 0

    tensor_keys = weights.keys() if use_mlx else sf.keys()
    for name in sorted(tensor_keys):
        gguf_name = map_tensor_name(name)
        if gguf_name is None:
            skipped += 1
            print(f"  SKIP: {name}")
            continue

        if use_mlx:
            tensor = weights[name]
            raw = np.array(mx.array(tensor, dtype=mx.float32))
        else:
            tensor = sf.get_tensor(name)
            raw = np.array(tensor)

        if raw.dtype == np.float16:
            data = raw.astype(np.float32)
        elif raw.dtype in (np.uint16, np.int16):
            # bf16 stored as uint16
            bf16_as_uint16 = raw.astype(np.uint16)
            fp32_bits = bf16_as_uint16.astype(np.uint32) << 16
            data = fp32_bits.view(np.float32)
        elif raw.dtype == np.float32:
            data = raw
        else:
            data = raw.astype(np.float32)

        total_size += data.nbytes

        # GGUFWriter records tensor dimensions in GGML order by reversing the
        # NumPy shape. Keep MLX/PyTorch [out, in] arrays as-is so llama.cpp sees
        # the expected [in, out] GGML dimensions.

        if len(data.shape) == 1 or "norm" in gguf_name:
            tensor_out = data.astype(np.float32)
            out_type = "f32"
        else:
            tensor_out = data.astype(np.float16)
            out_type = "f16"

        print(f"  ADD: {name} -> {gguf_name} {data.shape} -> {out_type} {tensor_out.nbytes/1e6:.1f}MB")
        writer.add_tensor(gguf_name, tensor_out)
        added += 1

    print(f"\nAdded: {added} tensors, Skipped: {skipped} tensors")
    print(f"Total uncompressed size: {total_size/1e9:.2f} GB")

    # Write the file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"\nDone! GGUF written to: {OUTPUT_PATH}")
    out_size = os.path.getsize(OUTPUT_PATH)
    print(f"File size: {out_size/1e9:.2f} GB")


if __name__ == "__main__":
    main()
