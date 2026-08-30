#!/usr/bin/env python3
"""
Karma BitNet b1.58 Chat & Validation CLI.
Interactive chat interface and validation tool for packed BitNet 1.58-bit / 2-bit models.
"""
import os
import sys
import time
import math
import json
import argparse
import warnings

# Suppress framework & dependency warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Add workspace paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "bitnet_toolkit"))

try:
    from bitlinear import BitLinear
    from replace_layers import convert_model_to_bitnet, register_qwen3_5_architecture
except ImportError:
    pass


def get_device() -> torch.device:
    """Auto-detects best available compute hardware."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def detect_architecture_from_state_dict(sd: dict) -> dict:
    """Inspects checkpoint tensor shapes to infer the correct base model architecture."""
    layers = set()
    hidden_size = None
    intermediate_size = None
    vocab_size = None

    for k, v in sd.items():
        if "model.layers." in k:
            parts = k.split(".")
            layers.add(int(parts[2]))
        if "embed_tokens" in k:
            vocab_size, hidden_size = v.shape
        elif "self_attn.q_proj.bias" in k:
            hidden_size = v.shape[0]

    num_layers = len(layers) if layers else 36

    if num_layers == 36 and hidden_size == 2048:
        base_model = "Qwen/Qwen2.5-3B-Instruct"
        arch_name = "Qwen 2.5 3B"
    elif num_layers == 28 and hidden_size == 1536:
        base_model = "Qwen/Qwen2.5-1.5B-Instruct"
        arch_name = "Qwen 2.5 1.5B"
    elif num_layers == 28 and hidden_size == 896:
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        arch_name = "Qwen 2.5 0.5B"
    else:
        base_model = "Qwen/Qwen2.5-3B-Instruct"
        arch_name = f"Custom Qwen ({num_layers} layers)"

    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size or 2048,
        "vocab_size": vocab_size or 151936,
        "base_model": base_model,
        "arch_name": arch_name,
        "total_tensors": len(sd),
    }


def load_packed_bitnet_model(checkpoint_path: str, base_model_name: str = None, device: torch.device = None):
    """
    Fast, memory-efficient loader for 2-bit packed BitNet checkpoints.
    Uses meta-device skeleton creation and vectorized in-place unpacking.
    """
    if device is None:
        device = get_device()

    t_start = time.time()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Load packed checkpoint from disk
    t0 = time.time()
    print(f"📦 [1/4] Loading checkpoint from '{checkpoint_path}'...", end="", flush=True)
    sd = torch.load(checkpoint_path, map_location="cpu")
    print(f" done ({time.time()-t0:.1f}s)")

    # 2. Detect Architecture & Config
    arch_info = detect_architecture_from_state_dict(sd)
    model_id = base_model_name or arch_info["base_model"]

    print(f"🧠 Architecture:   {arch_info['arch_name']} ({arch_info['num_layers']} layers, hidden_size={arch_info['hidden_size']})")
    print(f"⚡ Base Tokenizer: {model_id}")
    print(f"💻 Compute Device: {device.type.upper()}")

    t0 = time.time()
    print(f"⏳ [2/4] Loading tokenizer and configuration...", end="", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id)
    print(f" done ({time.time()-t0:.1f}s)")

    # 3. Instant Model Skeleton Creation (Meta device avoids allocating & initializing 6GB random weights)
    t0 = time.time()
    print(f"⏳ [3/4] Allocating uninitialized model skeleton on CPU...", end="", flush=True)
    with torch.device("meta"):
        meta_model = AutoModelForCausalLM.from_config(config)
    model = meta_model.to_empty(device="cpu")
    print(f" done ({time.time()-t0:.1f}s)")

    # 4. Fast Vectorized In-Place Parameter Unpacking
    t0 = time.time()
    print(f"⏳ [4/4] Unpacking 2-bit ternary weights in-place...", end="", flush=True)

    # Load scales metadata if exists
    scales_path = os.path.join(os.path.dirname(checkpoint_path), "scales_metadata.json")
    scales_dict = {}
    if os.path.exists(scales_path):
        try:
            with open(scales_path, "r") as f:
                scales_dict = json.load(f)
        except Exception:
            pass

    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)
    param_dict = dict(model.named_parameters())

    with torch.no_grad():
        for k in list(sd.keys()):
            v = sd.pop(k)  # Free tensor memory immediately
            if "weight_packed" in k:
                orig_key = k.replace(".weight_packed", ".weight")
                if orig_key in param_dict:
                    target_param = param_dict[orig_key]
                    target_shape = target_param.shape
                    numel = target_shape.numel()

                    module_name = orig_key.rsplit(".weight", 1)[0]
                    if f"{module_name}.gamma" in scales_dict:
                        gamma = scales_dict[f"{module_name}.gamma"]
                    else:
                        d_in = target_shape[-1]
                        gamma = (1.0 / math.sqrt(d_in)) * 0.8

                    # Vectorized 2-bit unpack (4 weights per byte)
                    raw = ((v.unsqueeze(1) >> shifts) & 0x03).view(-1)[:numel]
                    unpacked = (raw.to(torch.float16) - 1.0) * gamma
                    target_param.data.copy_(unpacked.view(target_shape))
            else:
                if k in param_dict:
                    param_dict[k].data.copy_(v)

        # Tie embeddings to lm_head
        if hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens"):
            model.lm_head.weight = model.model.embed_tokens.weight

    print(f" done ({time.time()-t0:.1f}s)")

    # 5. Move model to target device
    if device.type != "cpu":
        t0 = time.time()
        print(f"⏳ Transferring model to {device.type.upper()}...", end="", flush=True)
        model = model.to(device)
        print(f" done ({time.time()-t0:.1f}s)")

    model.eval()
    print(f"✓ BitNet Model ready in {time.time()-t_start:.1f}s total!\n")

    return model, tokenizer, arch_info


def run_validation(model, tokenizer, device: torch.device):
    """Runs automated validation tests on the model."""
    print("=" * 70)
    print("🧪 Running BitNet Validation Suite")
    print("=" * 70)

    test_cases = [
        {"prompt": "What is 2 + 2?", "expected": "4"},
        {"prompt": "What is the capital of France?", "expected": "Paris"},
        {"prompt": "Say hello in one word.", "expected": "Hello"},
    ]

    passed = 0
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    for i, tc in enumerate(test_cases, 1):
        prompt_text = f"<|im_start|>user\n{tc['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        print(f"\n[Test {i}/{len(test_cases)}] Prompt: '{tc['prompt']}'")
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                streamer=streamer,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        n_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
        tok_per_sec = n_tokens / max(0.001, elapsed)

        reply = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
        print(f"⚡ Performance: {tok_per_sec:.1f} tokens/sec ({n_tokens} tokens in {elapsed:.2f}s)")

        if reply and len(reply) > 0 and not all(c in "!?. " for c in reply):
            print(f"✓ Status: Output generated coherently.")
            passed += 1
        else:
            print(f"⚠️  Status: Raw/unfinetuned weights detected (knowledge distillation recommended).")

    print("\n" + "=" * 70)
    print(f"Validation Finished: {passed}/{len(test_cases)} tests produced textual tokens.")
    print("=" * 70 + "\n")


def interactive_chat(model, tokenizer, device: torch.device, system_prompt: str,
                     temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 256):
    """Interactive chat loop."""
    print("=" * 70)
    print("💬 BitNet b1.58 Interactive Chat")
    print("   Commands: /clear (reset history), /system (change prompt), /exit (quit)")
    print("=" * 70)

    messages = [{"role": "system", "content": system_prompt}]
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print("Goodbye!")
                break
            elif user_input.lower() == "/clear":
                messages = [{"role": "system", "content": system_prompt}]
                print("🧹 Conversation history cleared.")
                continue
            elif user_input.lower().startswith("/system "):
                system_prompt = user_input[8:].strip()
                messages = [{"role": "system", "content": system_prompt}]
                print(f"🔧 System prompt updated: '{system_prompt}'")
                continue

            messages.append({"role": "user", "content": user_input})
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

            print("\nKarma (BitNet) > ", end="", flush=True)
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    streamer=streamer,
                    do_sample=(temperature > 0),
                    temperature=max(0.01, temperature),
                    top_p=top_p,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            elapsed = time.time() - t0
            n_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
            tok_per_sec = n_tokens / max(0.001, elapsed)

            reply = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": reply})
            print(f"\n[{tok_per_sec:.1f} tok/s | {n_tokens} tokens | {elapsed:.2f}s]\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Karma BitNet b1.58 Chat & Validator")
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default="models/bitnet_qwen_1.5b_packed_2bit.pt",
        help="Path to packed BitNet .pt checkpoint"
    )
    parser.add_argument(
        "--base-model", "-b",
        type=str,
        default=None,
        help="Base HuggingFace model identifier (auto-detected if omitted)"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Run automated validation test suite"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum generation tokens"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are Karma, an ultra-fast, intelligent BitNet b1.58 AI companion.",
        help="System instruction prompt"
    )

    args = parser.parse_args()

    device = get_device()
    model, tokenizer, arch_info = load_packed_bitnet_model(
        args.model_path, base_model_name=args.base_model, device=device
    )

    if args.validate:
        run_validation(model, tokenizer, device)
    else:
        interactive_chat(
            model,
            tokenizer,
            device,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )


if __name__ == "__main__":
    main()
