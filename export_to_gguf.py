#!/usr/bin/env python3
"""
Automated BitNet to GGUF Exporter.
Converts BitNet 2-bit packed .pt checkpoints into GGUF format for bitnet.cpp and llama.cpp.
"""
import os
import sys
import json
import argparse
import subprocess
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

def export_checkpoint_to_gguf(
    checkpoint_path: str = "models/bitnet_qwen_1.5b_packed_2bit.pt",
    scales_path: str = "models/scales_metadata.json",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "models/bitnet_qwen_hf",
    output_gguf: str = "models/bitnet_qwen_3b_q2_k.gguf"
):
    print("=" * 65)
    print("📦 BitNet Checkpoint -> GGUF Converter")
    print("=" * 65)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/4] Reading packed weights and scales metadata...")
    sd = torch.load(checkpoint_path, map_location="cpu")
    
    scales = {}
    if os.path.exists(scales_path):
        with open(scales_path) as f:
            scales = json.load(f)
        print(f"✓ Loaded {len(scales)} layer scale factors from '{scales_path}'")
    else:
        print(f"⚠️ Scales file '{scales_path}' not found. Using default gamma estimation.")

    print(f"[2/4] Downloading architecture config for '{model_name}'...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"[3/4] Unpacking 2-bit ternary buffers into continuous tensors...")
    unpacked_tensors = {}
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads

    for k in list(sd.keys()):
        v = sd.pop(k)
        if "weight_packed" in k:
            orig_key = k.replace(".weight_packed", ".weight")
            mod_name = orig_key.rsplit(".weight", 1)[0]
            gamma = scales.get(f"{mod_name}.gamma", 0.02)

            if "q_proj" in orig_key or "o_proj" in orig_key:
                target_shape = (hidden_size, hidden_size)
            elif "k_proj" in orig_key or "v_proj" in orig_key:
                target_shape = (num_kv_heads * head_dim, hidden_size)
            elif "gate_proj" in orig_key or "up_proj" in orig_key:
                target_shape = (intermediate_size, hidden_size)
            elif "down_proj" in orig_key:
                target_shape = (hidden_size, intermediate_size)
            else:
                continue

            numel = target_shape[0] * target_shape[1]
            raw = ((v.unsqueeze(1) >> shifts) & 0x03).view(-1)[:numel]
            unpacked = (raw.to(torch.float16) - 1.0) * gamma
            unpacked_tensors[orig_key] = unpacked.view(target_shape).contiguous()
        else:
            unpacked_tensors[k] = v.to(torch.float16 if v.is_floating_point() else v.dtype).contiguous()

    if "lm_head.weight" not in unpacked_tensors and "model.embed_tokens.weight" in unpacked_tensors:
        unpacked_tensors["lm_head.weight"] = unpacked_tensors["model.embed_tokens.weight"].clone().contiguous()

    safetensors_file = os.path.join(output_dir, "model.safetensors")
    save_file(unpacked_tensors, safetensors_file)
    print(f"✓ Saved Hugging Face model directory to '{output_dir}'")

    print(f"[4/4] Converting to GGUF format and quantizing...")
    f16_gguf = output_gguf.replace(".gguf", "_f16.gguf")
    convert_script = "bitnet_cpp/3rdparty/llama.cpp/convert_hf_to_gguf.py"
    
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"GGUF convert script not found at '{convert_script}'. Run 'git submodule update --init --recursive' in bitnet_cpp.")

    cmd_convert = [
        sys.executable, convert_script,
        output_dir,
        "--outfile", f16_gguf,
        "--outtype", "f16"
    ]
    subprocess.run(cmd_convert, check=True)

    quant_bin = "bitnet_cpp/build/bin/llama-quantize"
    if os.path.exists(quant_bin):
        print(f"Applying 2-bit quantization with {quant_bin}...")
        cmd_quant = [quant_bin, f16_gguf, output_gguf, "Q2_K", "4"]
        subprocess.run(cmd_quant, check=True)
        if os.path.exists(f16_gguf):
            os.remove(f16_gguf)
        print(f"\n🎉 Successfully exported BitNet GGUF: {output_gguf} (Size: {os.path.getsize(output_gguf)/(1024**3):.2f} GB)")
    else:
        print(f"✓ GGUF saved at '{f16_gguf}' (Compile bitnet_cpp to enable 2-bit compression)")

def main():
    parser = argparse.ArgumentParser(description="Export BitNet Checkpoint to GGUF")
    parser.add_argument("--checkpoint", type=str, default="models/bitnet_qwen_1.5b_packed_2bit.pt")
    parser.add_argument("--scales", type=str, default="models/scales_metadata.json")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out-dir", type=str, default="models/bitnet_qwen_hf")
    parser.add_argument("--out-gguf", type=str, default="models/bitnet_qwen_3b_q2_k.gguf")

    args = parser.parse_args()
    export_checkpoint_to_gguf(
        checkpoint_path=args.checkpoint,
        scales_path=args.scales,
        model_name=args.model_name,
        output_dir=args.out_dir,
        output_gguf=args.out_gguf
    )

if __name__ == "__main__":
    main()
