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
with config.SilenceStderrFD():
    try:
        import torch
    except ImportError:
        pass
from src.cognition.engine import clean_companion_reply
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
        {"prompt": "Explain in two sentences why the sky is blue.", "expected": "blue"},
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
        first_dt = None

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
                if first_dt is None:
                    first_dt = time.time() - t0
                txt = chunk["choices"][0]["text"]
                print(txt, end="", flush=True)
                tokens.append(txt)

        total = time.time() - t0
        first_dt = first_dt if first_dt is not None else total
        decode_dt = max(total - first_dt, 0.001)
        reply = "".join(tokens).strip()
        try:
            n_tok = len(llm.tokenize(reply.encode("utf-8")))
        except Exception:
            n_tok = len(tokens)
        decode_tps = n_tok / decode_dt
        print(f"\n   TTFT: {first_dt*1000:.0f}ms | decode: {decode_tps:.1f} tok/s ({n_tok} tokens in {decode_dt:.2f}s, {total:.2f}s total)")

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
                        print("Indexed Documents:")
                        for d in docs:
                            print(f"  • {d['source']}: {d['count']} chunks")
                else:
                    print("RAG engine not loaded.")
                continue
            elif user_input.lower().startswith("/pdf "):
                pdf_target = user_input[5:].strip()
                if not rag_engine:
                    with config.SilenceStderrFD():
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

            messages.append({"role": "user", "content": user_input})

            prompt = ""
            for i, m in enumerate(messages):
                content = m["content"]
                if i == 0 and doc_context:
                    content = (
                        f"{content}\n\n"
                        f"Relevant Reference Knowledge:\n{doc_context}\n\n"
                        f"(Important: Use the reference knowledge only if it directly answers the user's question. Answer strictly in the same language as the user's message: if the user asks in English, reply ONLY in English; if the user asks in Arabic, reply ONLY in Egyptian Arabic. Never output multiple languages or translation notes.)"
                    )
                prompt += f"<|im_start|>{m['role']}\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

            t0 = time.time()
            first_token_time = None
            tokens = []

            if hasattr(llm, "stream_chat"):
                effective_sys = system_prompt
                if doc_context:
                    effective_sys = (
                        f"{system_prompt}\n\n"
                        f"Relevant Reference Knowledge:\n{doc_context}\n\n"
                        f"(Important: Use the reference knowledge only if it directly answers the user's question. Answer strictly in the same language as the user's message: if the user asks in English, reply ONLY in English; if the user asks in Arabic, reply ONLY in Egyptian Arabic. Never output multiple languages or translation notes.)"
                    )
                stream = llm.stream_chat(effective_sys, user_input, max_tokens=max_tokens)
                for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.time() - t0
                    tokens.append(chunk)
            else:
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
                        tokens.append(chunk["choices"][0]["text"])

            elapsed = time.time() - t0
            ttft = first_token_time or elapsed
            decode_dt = max(elapsed - ttft, 0.001)
            n_tokens = len(tokens)
            try:
                n_tok_true = len(llm.tokenize("".join(tokens).encode("utf-8")))
            except Exception:
                n_tok_true = n_tokens
            decode_tps = n_tok_true / decode_dt
            ttft_ms = ttft * 1000

            raw_reply = "".join(tokens).strip()
            reply = clean_companion_reply(raw_reply, user_input=user_input)
            print("\nKarma > ", end="", flush=True)
            for ch in reply:
                print(ch, end="", flush=True)
                time.sleep(0.005)

            messages.append({"role": "assistant", "content": reply})

            print(f"\n\033[90m[{decode_tps:.1f} tok/s decode | TTFT: {ttft_ms:.0f}ms | {n_tok_true} tokens | {elapsed:.2f}s total]\033[0m\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Karma Brain Interactive CLI")
    parser.add_argument("--model", "-m", type=str, default=get_default_model_path(), help="Path to .gguf model")
    parser.add_argument("--groq", "-g", action="store_true", help="Use Groq cloud API instead of local GGUF")
    parser.add_argument("--groq-model", type=str, default=getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"), help="Groq model name")
    parser.add_argument("--ctx-size", "-c", type=int, default=getattr(config, "CTX_SIZE", 4096), help="Context window")
    parser.add_argument("--threads", "-t", type=int, default=getattr(config, "N_THREADS", 4), help="CPU thread count")
    parser.add_argument("--temperature", type=float, default=getattr(config, "DEFAULT_TEMPERATURE", 0.7), help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=getattr(config, "DEFAULT_TOP_P", 0.9), help="Top-p nucleus sampling")
    parser.add_argument("--repeat-penalty", type=float, default=getattr(config, "DEFAULT_REPEAT_PENALTY", 1.05), help="Repetition penalty (1.05 recommended for Qwen 2.5)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max response tokens")
    parser.add_argument("--pdf", "-p", type=str, default=None, help="Path to PDF document to ingest into RAG before chat")
    parser.add_argument("--validate", "-v", action="store_true", help="Run automated validation test suite")
    parser.add_argument("--no-rag", action="store_true", help="Disable document RAG retrieval")
    parser.add_argument("--system-prompt", "-s", type=str, default=getattr(config, "PERSONA_SYSTEM_PROMPT", "You are Karma, a witty human friend, when someone greets like saying 'Hi', or 'Hello' you should respond with the same greeting"), help="System persona")

    args = parser.parse_args()

    rag_engine = None
    if not args.no_rag:
        try:
            with config.SilenceStderrFD():
                from src.memory.rag import DocumentRAG
                rag_engine = DocumentRAG()
            if args.pdf:
                rag_engine.ingest_pdf(args.pdf)
        except Exception as e:
            config.log_debug(f"[chat] RAG init note: {e}")

    if args.groq:
        from src.cognition.engine import create_engine
        print("=" * 65)
        print(f"⚡ Karma Brain — Groq Cloud Engine ({args.groq_model})")
        print("=" * 65)
        llm = create_engine(use_groq=True, model_name=args.groq_model)
    else:
        llm = load_engine(args.model, ctx_size=args.ctx_size, threads=args.threads)

    if args.validate:
        if args.groq:
            print("Validation suite runs with local GGUF engine.")
        else:
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
