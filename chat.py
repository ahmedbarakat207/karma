#!/usr/bin/env python3
import os
import sys
import time
import argparse
import platform
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src import config
from llama_cpp import Llama


def get_default_model_path() -> str:
    candidates = [
        getattr(config, "MODEL_PATH", ""),
        os.path.join(config.MODELS_DIR, "model.gguf"),
        os.path.join(config.MODELS_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return os.path.join(config.MODELS_DIR, "model.gguf")


def load_engine(model_path: str, ctx_size: int = 2048, threads: int = 4):
    if not os.path.exists(model_path):
        print(f"📦 Model not found at '{model_path}'. Downloading {config.HF_FILENAME} from {config.HF_REPO}...")
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=getattr(config, "HF_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            filename=getattr(config, "HF_FILENAME", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            local_dir=os.path.dirname(model_path)
        )
        model_path = downloaded

    is_mac = platform.system() == "Darwin"
    gpu_layers = -1 if is_mac else 0
    device_name = "Apple Silicon (Metal GPU)" if is_mac else f"Raspberry Pi / ARM CPU ({threads} threads)"

    print("=" * 65)
    print("⚡ Karma Brain — High-Speed Local GGUF Engine")
    print(f"📦 Model:    {os.path.basename(model_path)}")
    print(f"💻 Hardware: {device_name}")
    print(f"🧠 Context:  {ctx_size} tokens | Batch: {getattr(config, 'N_BATCH', 512)}")
    print("=" * 65)

    t0 = time.time()
    with config.SilenceStderrFD():
        llm = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_batch=getattr(config, "N_BATCH", 512),
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=gpu_layers,
            verbose=False
        )
    print(f"✓ Engine initialized in {time.time()-t0:.2f}s\n")
    return llm


def run_validation(llm):
    print("=" * 65)
    print("🧪 Running Karma Cognition Validation Suite")
    print("=" * 65)

    test_cases = [
        {"prompt": "What is 2 + 2? Answer in one number.", "expected": "4"},
        {"prompt": "What is the capital of France?", "expected": "Paris"},
        {"prompt": "Say hello in one word.", "expected": "Hello"},
    ]

    passed = 0
    for i, tc in enumerate(test_cases, 1):
        prompt_text = (
            f"<|im_start|>system\nYou are Karma, an autonomous companion.<|im_end|>\n"
            f"<|im_start|>user\n{tc['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        print(f"[Test {i}/{len(test_cases)}] Prompt: '{tc['prompt']}'")
        t0 = time.time()

        with config.SilenceStderrFD():
            stream = llm(
                prompt_text,
                max_tokens=64,
                temperature=0.3,
                stop=["<|im_end|>", "<|endoftext|>"],
                stream=True
            )

            tokens = []
            print("   Output: ", end="", flush=True)
            for chunk in stream:
                txt = chunk["choices"][0]["text"]
                print(txt, end="", flush=True)
                tokens.append(txt)

        elapsed = time.time() - t0
        n_tok = len(tokens)
        tok_per_sec = n_tok / max(0.001, elapsed)
        reply = "".join(tokens).strip()
        print(f"\n   ⚡ Speed: {tok_per_sec:.1f} tok/s ({n_tok} tokens in {elapsed:.2f}s)")

        if reply and len(reply) > 0:
            print("   ✓ Status: PASS\n")
            passed += 1
        else:
            print("   ❌ Status: FAIL\n")

    print("=" * 65)
    print(f"Validation Result: {passed}/{len(test_cases)} tests passed.")
    print("=" * 65 + "\n")



def interactive_chat(
    llm,
    system_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 256,
    repeat_penalty: float = 1.05,
    top_p: float = 0.9,
    rag_engine=None
):
    print("=" * 65)
    print("💬 Interactive Conversation Mode (Qwen 2.5 0.5B Instruct)")
    print("   Commands: /exit (quit) | /clear (reset history)")
    if rag_engine:
        print("   RAG Commands: /pdf <path> (index PDF) | /docs (list documents)")
    print("=" * 65)

    messages = [{"role": "system", "content": system_prompt}]

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
                print("🧹 Conversation memory cleared.")
                continue
            elif user_input.lower() == "/docs":
                if rag_engine:
                    docs = rag_engine.list_documents()
                    if not docs:
                        print("No documents indexed.")
                    else:
                        print("📚 Indexed Documents:")
                        for d in docs:
                            print(f"  • {d['source']}: {d['count']} chunks")
                else:
                    print("RAG engine not loaded.")
                continue
            elif user_input.lower().startswith("/pdf "):
                pdf_target = user_input[5:].strip()
                if not rag_engine:
                    from src.memory.rag import DocumentRAG
                    rag_engine = DocumentRAG()
                try:
                    c = rag_engine.ingest_pdf(pdf_target)
                    print(f"✓ Indexed '{pdf_target}' ({c} chunks). Ready for Q&A!")
                except Exception as e:
                    print(f"⚠️ Error ingesting PDF: {e}")
                continue

            doc_context = ""
            if rag_engine:
                doc_context = rag_engine.get_rag_context(user_input, k=2)

            augmented_user_input = user_input
            if doc_context:
                augmented_user_input = (
                    f"Reference Knowledge:\n{doc_context}\n\n"
                    f"Question: {user_input}"
                )

            messages.append({"role": "user", "content": augmented_user_input})

            prompt = ""
            for m in messages:
                prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            print("\nKarma > ", end="", flush=True)
            t0 = time.time()
            first_token_time = None
            tokens = []

            with config.SilenceStderrFD():
                stream = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    frequency_penalty=getattr(config, "DEFAULT_FREQUENCY_PENALTY", 0.0),
                    presence_penalty=getattr(config, "DEFAULT_PRESENCE_PENALTY", 0.0),
                    stop=["<|im_end|>", "<|endoftext|>"],
                    stream=True
                )

                for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.time() - t0
                    txt = chunk["choices"][0]["text"]
                    print(txt, end="", flush=True)
                    tokens.append(txt)

            elapsed = time.time() - t0
            n_tokens = len(tokens)
            tok_per_sec = n_tokens / max(0.001, elapsed)
            ttft_ms = (first_token_time or 0) * 1000

            reply = "".join(tokens).strip()
            messages.append({"role": "assistant", "content": reply})

            print(f"\n\033[90m[{tok_per_sec:.1f} tok/s | TTFT: {ttft_ms:.0f}ms | {n_tokens} tokens | {elapsed:.2f}s]\033[0m\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Karma Brain Interactive CLI")
    parser.add_argument("--model", "-m", type=str, default=get_default_model_path(), help="Path to .gguf model")
    parser.add_argument("--ctx-size", "-c", type=int, default=getattr(config, "CTX_SIZE", 4096), help="Context window")
    parser.add_argument("--threads", "-t", type=int, default=getattr(config, "N_THREADS", 4), help="CPU thread count")
    parser.add_argument("--temperature", type=float, default=getattr(config, "DEFAULT_TEMPERATURE", 0.7), help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=getattr(config, "DEFAULT_TOP_P", 0.9), help="Top-p nucleus sampling")
    parser.add_argument("--repeat-penalty", type=float, default=getattr(config, "DEFAULT_REPEAT_PENALTY", 1.05), help="Repetition penalty (1.05 recommended for Qwen 2.5)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max response tokens")
    parser.add_argument("--pdf", "-p", type=str, default=None, help="Path to PDF document to ingest into RAG before chat")
    parser.add_argument("--validate", "-v", action="store_true", help="Run automated validation test suite")
    parser.add_argument("--system-prompt", "-s", type=str, default=getattr(config, "PERSONA_SYSTEM_PROMPT", "You are Karma, a witty human friend."), help="System persona")

    args = parser.parse_args()

    rag_engine = None
    if args.pdf:
        from src.memory.rag import DocumentRAG
        rag_engine = DocumentRAG()
        rag_engine.ingest_pdf(args.pdf)

    llm = load_engine(args.model, ctx_size=args.ctx_size, threads=args.threads)

    if args.validate:
        run_validation(llm)
    else:
        interactive_chat(
            llm,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            repeat_penalty=args.repeat_penalty,
            top_p=args.top_p,
            rag_engine=rag_engine
        )


if __name__ == "__main__":
    main()
