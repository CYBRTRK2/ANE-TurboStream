#!/usr/bin/env python3
"""Build a llama.cpp-loadable DFlash draft GGUF by raw-copying shared head tensors.

The z-lab DFlash draft omits token embeddings and LM head tensors because the
reference runtime shares them with the target model. llama.cpp model loading
requires those tensors to exist. This script preserves the draft GGUF metadata
and tensors, then appends the target model's raw `token_embd.weight` and
`output.weight` tensors without dequantizing them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/Desktop/ANE project/vendor/anemll-flash-llama.cpp/gguf-py"))

import gguf
from gguf import GGUFReader, GGUFValueType, GGUFWriter


DEFAULT_DRAFT = "~/models/Qwen3.5-35B-A3B-Draft-f16-tokenizerfix.gguf"
DEFAULT_TARGET = "~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
DEFAULT_OUTPUT = "~/models/Qwen3.5-35B-A3B-Draft-f16-loadable-rawhead.gguf"


def field_contents(field):
    value = field.contents()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        value = field_contents(alignment)
        if value is not None:
            writer.data_alignment = int(value)

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue

        value = field_contents(field)
        if value is None:
            continue

        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, value, value_type, sub_type=sub_type)


def add_reader_tensor(writer: GGUFWriter, reader: GGUFReader, name: str) -> None:
    tensor = next((t for t in reader.tensors if t.name == name), None)
    if tensor is None:
        raise RuntimeError(f"missing required tensor {name!r}")

    writer.add_tensor(
        tensor.name,
        tensor.data,
        raw_shape=tensor.data.shape,
        raw_dtype=tensor.tensor_type,
        tensor_endianess=reader.endianess,
    )
    print(f"  + {name}: logical_shape={list(tensor.shape)} raw_shape={tensor.data.shape} raw_type={tensor.tensor_type}")


def main() -> int:
    draft_path = Path(os.path.expanduser(os.environ.get("DFLASH_DRAFT_IN", DEFAULT_DRAFT)))
    target_path = Path(os.path.expanduser(os.environ.get("DFLASH_TARGET_IN", DEFAULT_TARGET)))
    output_path = Path(os.path.expanduser(os.environ.get("DFLASH_DRAFT_OUT", DEFAULT_OUTPUT)))

    if not draft_path.exists():
        raise FileNotFoundError(draft_path)
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    if output_path.exists() and os.environ.get("DFLASH_OVERWRITE") != "1":
        raise FileExistsError(f"{output_path} exists; set DFLASH_OVERWRITE=1 to replace it")

    print(f"Draft:  {draft_path}")
    print(f"Target: {target_path}")
    print(f"Output: {output_path}")

    draft = GGUFReader(str(draft_path))
    target = GGUFReader(str(target_path))

    draft_names = {t.name for t in draft.tensors}
    missing = [name for name in ("token_embd.weight", "output.weight") if name not in draft_names]
    if not missing:
        print("Draft already contains token_embd.weight and output.weight; nothing to do.")
        return 0
    print(f"Missing from draft: {', '.join(missing)}")

    arch_field = draft.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = field_contents(arch_field) if arch_field is not None else "qwen3"

    writer = GGUFWriter(str(output_path), arch=arch, endianess=draft.endianess)
    copy_metadata(draft, writer)

    print("Copying existing draft tensors...")
    for tensor in draft.tensors:
        writer.add_tensor(
            tensor.name,
            tensor.data,
            raw_shape=tensor.data.shape,
            raw_dtype=tensor.tensor_type,
            tensor_endianess=draft.endianess,
        )

    print("Appending shared tensors from target...")
    add_reader_tensor(writer, target, "token_embd.weight")
    add_reader_tensor(writer, target, "output.weight")

    writer.open_output_file(output_path)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"Done: {output_path} ({output_path.stat().st_size / 1024**3:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
