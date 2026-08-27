"""
BitNet b1.58 Model Exporter & Packer.
Packs trained ternary weights into 2-bit byte arrays (4 weights per uint8 byte)
for ultra-compact disk storage (~290 MB) and fast integer inference.
"""
import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bitlinear import BitLinear
from replace_layers import convert_model_to_bitnet, register_qwen3_5_architecture


def export_packed_bitnet(checkpoint_path: str, model_name: str, output_dir: str):
    print("=" * 70)
    print(f"📦 Packing BitNet b1.58 Weights: {checkpoint_path}")
    print("=" * 70)

    register_qwen3_5_architecture()
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/3] Instantiating base model architecture '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    from train_distill import load_any_causal_model
    model = load_any_causal_model(model_name, dtype=torch.float16)
    model, _, _ = convert_model_to_bitnet(model, verbose=False)



    print(f"[2/3] Loading fine-tuned weights from '{checkpoint_path}'...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    print(f"[3/3] Packing ternary weights into 2-bit buffers (4 weights/byte)...")
    packed_state_dict = {}
    scales_dict = {}

    total_orig_bytes = 0
    total_packed_bytes = 0

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            packed_weights, gamma = module.pack_ternary_weights()
            packed_state_dict[f"{name}.weight_packed"] = packed_weights.cpu()
            scales_dict[f"{name}.gamma"] = float(gamma.cpu().item())

            if module.bias is not None:
                packed_state_dict[f"{name}.bias"] = module.bias.data.cpu().to(torch.float16)

            orig_bytes = module.weight.numel() * 2  # FP16
            packed_bytes = packed_weights.numel()   # 2-bit
            total_orig_bytes += orig_bytes
            total_packed_bytes += packed_bytes

    # Copy non-ternary parameters (embeddings, norms, lm_head)
    for name, param in model.named_parameters():
        is_ternary = False
        for mod_name, mod in model.named_modules():
            if isinstance(mod, BitLinear) and name.startswith(mod_name):
                is_ternary = True
                break
        if not is_ternary:
            packed_state_dict[name] = param.data.cpu().to(torch.float16)
            total_orig_bytes += param.numel() * 2
            total_packed_bytes += param.numel() * 2

    # Save packed model
    packed_file = os.path.join(output_dir, "bitnet_qwen_1.5b_packed_2bit.pt")
    torch.save(packed_state_dict, packed_file)

    # Save scale metadata
    meta_file = os.path.join(output_dir, "scales_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(scales_dict, f, indent=2)

    # Save tokenizer & HF config
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)

    orig_mb = total_orig_bytes / (1024 * 1024)
    packed_mb = os.path.getsize(packed_file) / (1024 * 1024)

    print("\n" + "=" * 70)
    print(f"✓ Model successfully packed and exported!")
    print(f"Original FP16 Model Size: ~{orig_mb:.1f} MB")
    print(f"Packed 2-Bit Model Size:   ~{packed_mb:.1f} MB  (Compression Ratio: {orig_mb / packed_mb:.2f}x)")
    print(f"Saved artifacts to: '{output_dir}'")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="BitNet b1.58 Exporter")
    parser.add_argument("--checkpoint", type=str, default="./bitnet_qwen_1.5b_output/bitnet_final.pt", help="Path to trained .pt weights")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model name")
    parser.add_argument("--output-dir", type=str, default="./bitnet_qwen_1.5b_packed", help="Export directory")
    args = parser.parse_args()

    export_packed_bitnet(args.checkpoint, args.model_name, args.output_dir)


if __name__ == "__main__":
    main()
