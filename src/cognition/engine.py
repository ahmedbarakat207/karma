
import os
import re
from typing import Generator, Optional, Union, Any


from src import config

_OPEN_TAGS = {"<think>": "</think>", "<thought>": "</thought>"}
_MAX_TAG_LEN = max(len(t) for t in list(_OPEN_TAGS.keys()) + list(_OPEN_TAGS.values()))


def _strip_thinking(text: str) -> str:
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
                config.log_debug(f"[llm] enabled Prompt-Lookup Speculative Decoding (ngram={ngram_size}, pred_tokens={num_pred})")
            except Exception as e:
                config.log_debug(f"[llm] speculative decoding init note: {e}")


        import llama_cpp
        type_k = llama_cpp.GGML_TYPE_Q8_0 if getattr(config, "KV_CACHE_TYPE", "q8_0") == "q8_0" else llama_cpp.GGML_TYPE_F16
        type_v = llama_cpp.GGML_TYPE_Q8_0 if getattr(config, "KV_CACHE_TYPE", "q8_0") == "q8_0" else llama_cpp.GGML_TYPE_F16
        flash_attn = getattr(config, "FLASH_ATTN", True)

        with config.SilenceStderrFD():
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
        if hasattr(self, "llm") and self.llm is not None:
            try:
                self.llm.close()
            except Exception:
                pass
            self.llm = None

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 160,
             temperature: float = 0.7) -> str:
        if self.llm is None:
            return ""
        prompt = self._format_prompt(system_prompt, user_prompt)
        with config.SilenceStderrFD():
            out = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=getattr(config, "DEFAULT_TOP_P", 0.90),
                repeat_penalty=getattr(config, "DEFAULT_REPEAT_PENALTY", 1.05),
                frequency_penalty=getattr(config, "DEFAULT_FREQUENCY_PENALTY", 0.0),
                presence_penalty=getattr(config, "DEFAULT_PRESENCE_PENALTY", 0.0),
                stop=self.stop_tokens
            )
        text = out["choices"][0]["text"].strip()
        return _strip_thinking(text).strip()

    def stream_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 160,
                    temperature: float = 0.7) -> Generator[str, None, None]:
        if self.llm is None:
            return
        prompt = self._format_prompt(system_prompt, user_prompt)

        def raw_tokens():
            with config.SilenceStderrFD():
                stream = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=getattr(config, "DEFAULT_TOP_P", 0.90),
                    repeat_penalty=getattr(config, "DEFAULT_REPEAT_PENALTY", 1.05),
                    frequency_penalty=getattr(config, "DEFAULT_FREQUENCY_PENALTY", 0.0),
                    presence_penalty=getattr(config, "DEFAULT_PRESENCE_PENALTY", 0.0),
                    stop=self.stop_tokens,
                    stream=True
                )
                for chunk in stream:
                    yield chunk["choices"][0]["text"]

        yield from _strip_thinking_from_stream(raw_tokens())



class GroqEngine:

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        raw = model_name or getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b")
        self.model = f"openai/{raw}" if raw in ("gpt-oss-20b", "gpt-oss-120b", "gpt-oss-safeguard-20b") else raw
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.client = None

        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key) if self.api_key else Groq()
            config.log_debug(f"[llm] Groq engine initialized with model: {self.model}")

        except Exception as e:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.api_key or os.environ.get("GROQ_API_KEY", "EMPTY")
                )
                config.log_debug(f"[llm] Groq OpenAI-compatible client initialized with model: {self.model}")
            except Exception as e2:
                config.log_debug(f"[llm] Groq client init note: {e2}")

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024,
             temperature: float = 0.7) -> str:
        if not self.client:
            config.log_debug("[groq] client not initialized (set GROQ_API_KEY)")
            return ""
        try:
            budget = max(max_tokens, 1024)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=budget,
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            return _strip_thinking(text).strip()
        except Exception as e:
            config.log_debug(f"[groq] chat error: {e}")
            try:
                # Fallback without max_completion_tokens for non-o1 models
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=budget,
                    temperature=temperature,
                )
                text = response.choices[0].message.content or ""
                return _strip_thinking(text).strip()
            except Exception as e2:
                config.log_debug(f"[groq] fallback error: {e2}")
                return ""

    def stream_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024,
                    temperature: float = 0.7) -> Generator[str, None, None]:
        if not self.client:
            config.log_debug("[groq] client not initialized (set GROQ_API_KEY)")
            return

        budget = max(max_tokens, 1024)

        def raw_tokens():
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=budget,
                    temperature=temperature,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = getattr(chunk.choices[0], "delta", None)
                        content = getattr(delta, "content", None) if delta else None
                        if content:
                            yield content
            except Exception as e:
                config.log_debug(f"[groq] stream error: {e}")
                try:
                    # Fallback with standard max_tokens
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=budget,
                        temperature=temperature,
                        stream=True
                    )
                    for chunk in stream:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = getattr(chunk.choices[0], "delta", None)
                            content = getattr(delta, "content", None) if delta else None
                            if content:
                                yield content
                except Exception as e2:
                    config.log_debug(f"[groq] stream fallback error: {e2}")

        yield from _strip_thinking_from_stream(raw_tokens())

    def close(self) -> None:
        pass



def create_engine(use_groq: Optional[bool] = None, model_name: Optional[str] = None) -> Union[LocalEngine, GroqEngine]:
    should_use_groq = getattr(config, "USE_GROQ", False) if use_groq is None else use_groq
    if should_use_groq:
        model = model_name or getattr(config, "GROQ_MODEL", "gpt-oss-20b")
        return GroqEngine(model_name=model)
    return LocalEngine()

