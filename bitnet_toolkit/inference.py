"""
Interactive Generation & Benchmark Script for BitNet b1.58 Qwen 2.5 1.5B.
Runs on Apple Silicon MPS, CUDA, or multi-threaded CPU.
"""
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bitlinear import BitLinear
from replace_layers import convert_model_to_bitnet, register_qwen3_5_architecture


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_interactive_inference(checkpoint_path: str, model_name: str):
    device = get_device()
    print("=" * 70)
    print(f"🧠 BitNet b1.58 Inference Engine (Device: {device.type.upper()})")
    print("=" * 70)

    register_qwen3_5_architecture()
    print(f"[1/2] Loading tokenizer and base architecture...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    from train_distill import load_any_causal_model
    model = load_any_causal_model(model_name, dtype=dtype)
    model, converted, _ = convert_model_to_bitnet(model, verbose=False)



    if checkpoint_path and checkpoint_path != "none":
        print(f"[2/2] Loading trained BitNet weights from '{checkpoint_path}'...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("[2/2] Running in zero-shot BitNet emulation mode...")

    model = model.to(device)
    model.eval()

    print("\n✓ Model ready! Type a prompt below or press Ctrl+C to exit.\n")

    test_prompts = [
        "Explain quantum computing in one short sentence.",
        "Hey! How is your day going?",
        "Write a quick python function to reverse a string."
    ]

    print("--- Running Quick Benchmarks ---")
    for prompt in test_prompts:
        print(f"\nUser: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0

        generated_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
        tok_per_sec = generated_tokens / max(0.001, elapsed)

        reply = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        print(f"BitNet: {reply.strip()}")
        print(f"⚡ Performance: {tok_per_sec:.1f} tokens/sec ({generated_tokens} tokens in {elapsed:.2f}s)")

    print("\n" + "=" * 70)
    print("Interactive Chat (Type 'quit' to exit):")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input or user_input.lower() in ("quit", "exit"):
                break

            messages = [{"role": "user", "content": user_input}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt").to(device)

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            elapsed = time.time() - t0

            generated_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
            tok_per_sec = generated_tokens / max(0.001, elapsed)

            reply = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            print(f"\nKarma (BitNet) > {reply.strip()}")
            print(f"[{tok_per_sec:.1f} tok/s]")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="BitNet b1.58 Inference Benchmark")
    parser.add_argument("--checkpoint", type=str, default="none", help="Path to trained .pt weights (or 'none')")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model name")
    args = parser.parse_args()

    run_interactive_inference(args.checkpoint, args.model_name)


if __name__ == "__main__":
    main()
