"""
Language Model Subsystem ("The Mind").
100% In-Process Offline GGUF Inference via llama-cpp-python.
"""
import os
import re
from typing import Generator, Optional

from src import config

_OPEN_TAGS = {"<think>": "</think>", "<thought>": "</thought>"}
_MAX_TAG_LEN = max(len(t) for t in list(_OPEN_TAGS.keys()) + list(_OPEN_TAGS.values()))


def _strip_thinking(text: str) -> str:
    """Removes complete or unclosed thinking blocks from full response strings."""
    if not text:
        return ""
    for open_t, close_t in _OPEN_TAGS.items():
        while open_t in text:
            if close_t in text:
                pattern = re.escape(open_t) + r".*?" + re.escape(close_t)
                text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
            else:
                text = text.split(open_t)[0].strip()
    return text


def _strip_thinking_from_stream(token_iter: Generator[str, None, None]) -> Generator[str, None, None]:
    """Filters <think>...</think> tags in real-time from a streaming token iterator."""
    buf = ""
    active_close_tag = None

    for token in token_iter:
        buf += token
        while True:
            if active_close_tag is None:
                earliest_idx = -1
                matched_open = None
                for open_t in _OPEN_TAGS:
                    idx = buf.find(open_t)
                    if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                        earliest_idx = idx
                        matched_open = open_t

                if earliest_idx == -1:
                    safe = buf[:-_MAX_TAG_LEN] if len(buf) > _MAX_TAG_LEN else ""
                    if safe:
                        yield safe
                        buf = buf[len(safe):]
                    break
                else:
                    if earliest_idx > 0:
                        yield buf[:earliest_idx]
                    buf = buf[earliest_idx + len(matched_open):]
                    active_close_tag = _OPEN_TAGS[matched_open]
            else:
                idx = buf.find(active_close_tag)
                if idx == -1:
                    discard_up_to = max(0, len(buf) - len(active_close_tag))
                    buf = buf[discard_up_to:]
                    break
                else:
                    buf = buf[idx + len(active_close_tag):]
                    active_close_tag = None

    if buf and active_close_tag is None:
        yield buf


class LocalEngine:
    """100% in-process offline GGUF inference via llama-cpp-python (0 servers, 0 internet)."""

    def __init__(self, model_path: Optional[str] = None):
        from llama_cpp import Llama
        self.model_path = model_path or getattr(config, "MODEL_PATH", "")

        if not os.path.exists(self.model_path):
            repo = getattr(config, "HF_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
            filename = getattr(config, "HF_FILENAME", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
            print(f"[llm] model file not found at {self.model_path} -- downloading {filename} from {repo}...")
            try:
                from huggingface_hub import hf_hub_download
                parent_dir = os.path.dirname(self.model_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                downloaded = hf_hub_download(repo_id=repo, filename=filename, local_dir=parent_dir)
                self.model_path = downloaded
                print(f"[llm] download complete: {self.model_path}")
            except Exception as e:
                raise FileNotFoundError(f"Could not load or download model: {e}")

        threads = getattr(config, "N_THREADS", 4)
        draft_model = None
        if getattr(config, "SPECULATIVE_DECODING", "prompt_lookup") == "prompt_lookup":
            try:
                from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                ngram_size = getattr(config, "SPECULATIVE_NGRAM_SIZE", 2)
                num_pred = getattr(config, "SPECULATIVE_NUM_PRED_TOKENS", 8)
                draft_model = LlamaPromptLookupDecoding(max_ngram_size=ngram_size, num_pred_tokens=num_pred)
                print(f"[llm] enabled Prompt-Lookup Speculative Decoding (ngram={ngram_size}, pred_tokens={num_pred})")
            except Exception as e:
                print(f"[llm] speculative decoding init note: {e}")

        import llama_cpp
        type_k = llama_cpp.GGML_TYPE_Q8_0 if getattr(config, "KV_CACHE_TYPE", "q8_0") == "q8_0" else llama_cpp.GGML_TYPE_F16
        type_v = llama_cpp.GGML_TYPE_Q8_0 if getattr(config, "KV_CACHE_TYPE", "q8_0") == "q8_0" else llama_cpp.GGML_TYPE_F16
        flash_attn = getattr(config, "FLASH_ATTN", True)

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=getattr(config, "CTX_SIZE", 2048),
            n_batch=getattr(config, "N_BATCH", 512),
            n_threads=threads,
            n_threads_batch=threads,
            n_gpu_layers=getattr(config, "N_GPU_LAYERS", -1),
            type_k=type_k,
            type_v=type_v,
            flash_attn=flash_attn,
            draft_model=draft_model,
            verbose=False,
        )
        self.stop_tokens = ["<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<start_of_turn>"]

        import atexit
        atexit.register(self.close)

    def close(self) -> None:
        """Cleanly free llama model and Metal resources."""
        if hasattr(self, "llm") and self.llm is not None:
            try:
                self.llm.close()
            except Exception:
                pass
            self.llm = None

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Formats into standard ChatML for Qwen 2.5."""
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 300,
             temperature: float = 0.7) -> str:
        if self.llm is None:
            return ""
        prompt = self._format_prompt(system_prompt, user_prompt)
        out = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=self.stop_tokens)
        text = out["choices"][0]["text"].strip()
        return _strip_thinking(text).strip()

    def stream_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 300,
                    temperature: float = 0.7) -> Generator[str, None, None]:
        if self.llm is None:
            return
        prompt = self._format_prompt(system_prompt, user_prompt)

        def raw_tokens():
            stream = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=self.stop_tokens, stream=True)
            for chunk in stream:
                yield chunk["choices"][0]["text"]

        yield from _strip_thinking_from_stream(raw_tokens())


def create_engine() -> LocalEngine:
    """Factory: Instantiates the in-process local llama_cpp engine."""
    return LocalEngine()
