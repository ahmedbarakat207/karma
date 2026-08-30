#!/usr/bin/env python3
"""
BitNet C++ Runner: Interactive Streaming Chat using compiled C++ Metal Engine.
Runs BitNet GGUF models at maximum hardware speed on Apple Silicon / CPU.
"""
import os
import sys
import subprocess
import argparse

DEFAULT_MODEL_GGUF = "models/bitnet_qwen_3b_q2_k.gguf"
CLI_BIN = "bitnet_cpp/build/bin/llama-cli"

def main():
    parser = argparse.ArgumentParser(description="BitNet C++ Inference Engine")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL_GGUF, help="Path to .gguf model")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, default=99, help="Number of layers to offload to Metal GPU")
    parser.add_argument("-c", "--ctx-size", type=int, default=4096, help="Context window size")
    parser.add_argument("-temp", "--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="One-shot prompt (if not specified, launches interactive chat)")

    args = parser.parse_args()

    if not os.path.exists(CLI_BIN):
        print(f"❌ BitNet C++ binary not found at '{CLI_BIN}'.")
        print("Please build it first with: cd bitnet_cpp && cmake --build build --target llama-cli -j 4")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"❌ Model GGUF not found at '{args.model}'.")
        print("Export your BitNet checkpoint to GGUF first using: python3 export_to_gguf.py")
        sys.exit(1)

    cmd = [
        CLI_BIN,
        "-m", args.model,
        "-t", str(args.threads),
        "-ngl", str(args.n_gpu_layers),
        "-c", str(args.ctx_size),
        "--temp", str(args.temperature),
        "--color",
    ]

    if args.prompt:
        cmd.extend(["-p", args.prompt])
    else:
        # Interactive chat mode
        cmd.extend([
            "-cnv",
            "--chat-template", "chatml",
            "-n", "512"
        ])

    print("=" * 65)
    print(f"⚡ Launching BitNet C++ Inference Engine (Metal GPU + ARM NEON)")
    print(f"📦 Model: {args.model}")
    print("=" * 65)
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nSession closed.")

if __name__ == "__main__":
    main()
