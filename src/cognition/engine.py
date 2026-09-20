
import os
import re
import sys
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Union


from src import config

# Global heavy-compute gate: LLM inference and TTS synthesis must never run
# concurrently. On Pi 4 both libraries (llama.cpp OpenMP + onnxruntime) spawn
# full thread pools; overlapping them oversubscribes the 4 cores ~3x, and the
# resulting current spikes brown out marginal PSUs (under-voltage -> SIGSEGV).
# TTS *playback* (paplay/aplay subprocess) stays concurrent — only synthesis
# and inference serialize. Must be re-entrant for nested engine calls.
_HEAVY_COMPUTE_LOCK = threading.RLock()


def heavy_compute():
    """Context manager acquiring the heavy-compute gate."""
    return _HEAVY_COMPUTE_LOCK

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


def clean_companion_reply(text: str, user_input: Optional[str] = None) -> str:
    if not text:
        return text
    orig = text
    while True:
        prev = text
        text = re.sub(r'^(?:friend said|user said|user|karma|friend)[!:,.\s\"-]*', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^(?:hello|hi|hey)(?: there)?[!,.]?\s*how can i (?:assist|help) you(?: today)?\??', "Hey! What's up?", text, flags=re.IGNORECASE).strip()
        text = re.sub(r'how can i (?:assist|help) you(?: today)?\??', "what's up?", text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^as (?:an? )?(?:ai|artificial intelligence|language model|human friend|friend|machine)[^.!?\n]*(?:[.,!?]|\b(?:but|however),?)\s*', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^i (?:do not|don\'t) have (?:personal )?(?:preferences|emotions|feelings)[^.!?\n]*(?:[.,!?]|\b(?:but|however),?)\s*', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'^(?:however|but),?\s*', '', text, flags=re.IGNORECASE).strip()
        if text == prev:
            break

    # Strip bilingual / translation leakage
    is_ar_user = any('\u0600' <= ch <= '\u06FF' for ch in user_input) if user_input else False
    if is_ar_user:
        m = re.search(r'\*?بالعامية(?: المصرية)?:\*?\s*(.+)$', text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    else:
        text = re.sub(r'\n+\s*\*?بالعامية(?: المصرية)?:\*?.*$', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\n+\s*(?:In Egyptian Arabic|Egyptian Arabic|Arabic translation):\s*.*$', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

    if text and text != orig:
        text = text[0].upper() + text[1:]
    return text


class LocalEngine:

    def __init__(self, model_path: Optional[str] = None):
        from llama_cpp import Llama
        self.model_path = model_path or getattr(config, "MODEL_PATH", "")

        if not os.path.exists(self.model_path):
            repo = getattr(config, "HF_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
            filename = getattr(config, "HF_FILENAME", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
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
                raise FileNotFoundError(f"Could not load or download model at {self.model_path}: {e}")

        threads = getattr(config, "N_THREADS", 4)
        draft_model = None
        spec_mode = str(getattr(config, "SPECULATIVE_DECODING", "none")).lower()
        is_arm_linux = sys.platform.startswith("linux") and (
            hasattr(os, "uname") and any(a in os.uname().machine.lower() for a in ("arm", "aarch"))
        )
        if spec_mode == "prompt_lookup" and not is_arm_linux:
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

        # Flash attention is only supported on CUDA or Apple Silicon MPS; never on ARM CPU
        gpu_device = getattr(config, "_DEFAULT_YOLO_DEVICE", "cpu")
        flash_attn = getattr(config, "FLASH_ATTN", False) if gpu_device in ("mps", "cuda") else False
        n_gpu_layers = getattr(config, "N_GPU_LAYERS", 0) if gpu_device in ("mps", "cuda") else 0

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=getattr(config, "CTX_SIZE", 2048),
                n_batch=getattr(config, "N_BATCH", 512),
                n_threads=threads,
                n_threads_batch=threads,
                n_gpu_layers=n_gpu_layers,
                type_k=type_k,
                type_v=type_v,
                flash_attn=flash_attn,
                draft_model=draft_model,
                verbose=False,
            )
        except Exception as e_init:
            config.log_debug(f"[llm] Llama initial load note ({e_init}), retrying with safe CPU defaults...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=min(getattr(config, "CTX_SIZE", 2048), 2048),
                n_batch=min(getattr(config, "N_BATCH", 512), 256),
                n_threads=threads,
                n_threads_batch=threads,
                n_gpu_layers=0,
                flash_attn=False,
                draft_model=None,
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

    def _format_prompt(self, system_prompt: str, user_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        if history:
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "").strip()
                if content:
                    prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 160,
             temperature: float = 0.7, history: Optional[List[Dict[str, str]]] = None) -> str:
        if self.llm is None:
            return ""
        # Collect via the streaming path so telemetry records an honest
        # TTFT + decode-only tok/s (the old non-streaming call lumped slow
        # prompt processing into "TPS"). Sampling params identical.
        # Serialized against TTS synthesis via the heavy-compute gate.
        parts: List[str] = []
        t0 = time.time()
        first = 0.0
        with heavy_compute():
            for tok in self.stream_chat(system_prompt, user_prompt, max_tokens=max_tokens,
                                        temperature=temperature, history=history):
                if not first:
                    first = time.time()
                parts.append(tok)
        text = "".join(parts).strip()
        try:
            from src.ui import telemetry as _telemetry
            # stream_chat already recorded TTFT/decode; ensure non-stream
            # callers without a running loop still get a sane entry.
            if not first:
                _telemetry.record_llm(time.time() - t0, time.time() - t0,
                                      max(1, len(text) // 4))
        except Exception:
            pass
        return clean_companion_reply(_strip_thinking(text).strip())

    def stream_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 160,
                    temperature: float = 0.7, history: Optional[List[Dict[str, str]]] = None) -> Generator[str, None, None]:
        if self.llm is None:
            return
        prompt = self._format_prompt(system_prompt, user_prompt, history=history)

        _t0 = time.time()
        _first = [0.0]
        _count = [0]

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
                    if not _first[0]:
                        _first[0] = time.time()
                    _count[0] += 1
                    yield chunk["choices"][0]["text"]
            try:
                from src.ui import telemetry as _telemetry
                _telemetry.record_llm((_first[0] or time.time()) - _t0,
                                      time.time() - _t0, max(1, _count[0]))
            except Exception:
                pass

        yield from _strip_thinking_from_stream(raw_tokens())



class GroqEngine:

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        raw = model_name or getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b")
        self.model = f"openai/{raw}" if raw in ("gpt-oss-20b", "gpt-oss-120b", "gpt-oss-safeguard-20b") else raw
        self.api_key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
        self.client = None
        self.timeout = float(getattr(config, "GROQ_TIMEOUT", 8.0))

        if not self.api_key:
            # Fail fast: no network attempt without a key. SwitchableEngine
            # falls back to local immediately instead of paying a timeout.
            config.log_debug("[groq] no GROQ_API_KEY — engine will return empty (fast fallback to local)")
            return

        try:
            from groq import Groq
            # Single short timeout: a dead network must not stall every
            # utterance, then pay a second full local inference as fallback.
            try:
                self.client = Groq(api_key=self.api_key, timeout=self.timeout)
            except TypeError:
                self.client = Groq(api_key=self.api_key)
            config.log_debug(f"[llm] Groq engine initialized with model: {self.model}")

        except Exception as e:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
                config.log_debug(f"[llm] Groq OpenAI-compatible client initialized with model: {self.model}")
            except Exception as e2:
                config.log_debug(f"[llm] Groq client init note: {e2}")

    def _build_groq_messages(self, system_prompt: str, user_prompt: str,
                             history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history:
                if turn.get("content"):
                    msgs.append({"role": turn.get("role", "user"), "content": turn["content"]})
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    def _has_key(self) -> bool:
        # Re-read env each call so dashboard key updates apply without restart.
        if self.api_key and self.api_key.strip():
            return True
        env_key = os.environ.get("GROQ_API_KEY", "").strip()
        if env_key:
            self.api_key = env_key
            return True
        return False

    def _create_kwargs(self, budget: int, temperature: float, stream: bool = False) -> Dict[str, Any]:
        # Prefer max_completion_tokens (new Groq API); fall back to max_tokens
        # only on TypeError (old SDK), never with a second network call.
        base: Dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
        }
        # gpt-oss reasoning models think out loud before answering — default
        # effort burns hundreds of hidden tokens (≈ seconds) while the robot
        # screen stays blank. "low" keeps answers instant; quality is fine
        # for 1-2 sentence companion replies.
        if "gpt-oss" in str(self.model):
            base["reasoning_effort"] = "low"
        if stream:
            base["stream"] = True
        return base

    def _budget(self, max_tokens: int) -> int:
        # gpt-oss reasoning models burn ~50 hidden reasoning tokens before
        # the answer — a tiny budget (e.g. 20-75) starves the answer and
        # returns empty. Groq bills caps, not targets (0.06s either way),
        # so give reasoning models headroom; others use the caller's budget.
        want = max(1, int(max_tokens))
        if "gpt-oss" in str(self.model):
            return max(want, 1024)
        return want

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 160,
             temperature: float = 0.7, history: Optional[List[Dict[str, str]]] = None) -> str:
        if not self.client or not self._has_key():
            config.log_debug("[groq] client not initialized (set GROQ_API_KEY)")
            return ""
        budget = self._budget(max_tokens)
        t0 = time.time()
        msgs = self._build_groq_messages(system_prompt, user_prompt, history=history)
        try:
            try:
                response = self.client.chat.completions.create(
                    **self._create_kwargs(budget, temperature),
                    messages=msgs,
                    max_completion_tokens=budget,
                    timeout=self.timeout,
                )
            except TypeError:
                # Old SDK without max_completion_tokens/timeout kwargs.
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=msgs,
                        max_tokens=budget,
                        temperature=temperature,
                    )
                except TypeError:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=msgs,
                        max_tokens=budget,
                        temperature=temperature,
                    )
            text = response.choices[0].message.content or ""
            try:
                from src.ui import telemetry as _telemetry
                usage = getattr(response, "usage", None)
                done = int(getattr(usage, "completion_tokens", 0) or max(1, len(text) // 4))
                _telemetry.record_llm(time.time() - t0, time.time() - t0, done)
            except Exception:
                pass
            return clean_companion_reply(_strip_thinking(text).strip())
        except Exception as e:
            # Single attempt only — no second network call. Caller
            # (SwitchableEngine) falls back to local immediately.
            config.log_debug(f"[groq] chat error: {e}")
            return ""

    def stream_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 160,
                    temperature: float = 0.7, history: Optional[List[Dict[str, str]]] = None) -> Generator[str, None, None]:
        if not self.client or not self._has_key():
            config.log_debug("[groq] client not initialized (set GROQ_API_KEY)")
            return

        budget = self._budget(max_tokens)
        groq_msgs = self._build_groq_messages(system_prompt, user_prompt, history=history)
        _t0 = time.time()
        _first = [0.0]
        _count = [0]

        def raw_tokens():
            try:
                try:
                    _stream_kwargs: Dict[str, Any] = {
                        "model": self.model,
                        "messages": groq_msgs,
                        "max_completion_tokens": budget,
                        "temperature": temperature,
                        "stream": True,
                        "timeout": self.timeout,
                    }
                    if "gpt-oss" in str(self.model):
                        _stream_kwargs["reasoning_effort"] = "low"
                    stream = self.client.chat.completions.create(**_stream_kwargs)
                except TypeError:
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=groq_msgs,
                        max_tokens=budget,
                        temperature=temperature,
                        stream=True,
                    )
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = getattr(chunk.choices[0], "delta", None)
                        content = getattr(delta, "content", None) if delta else None
                        if content:
                            if not _first[0]:
                                _first[0] = time.time()
                            _count[0] += 1
                            yield content
            except Exception as e:
                # Single attempt only — no second stream (was doubling tail latency).
                config.log_debug(f"[groq] stream error: {e}")
                return

        for tok in _strip_thinking_from_stream(raw_tokens()):
            yield tok
        try:
            from src.ui import telemetry as _telemetry
            _telemetry.record_llm((_first[0] or time.time()) - _t0,
                                  time.time() - _t0, max(1, _count[0]))
        except Exception:
            pass

    def close(self) -> None:
        pass



class SwitchableEngine:
    """A thread-safe engine wrapper that can dynamically switch between
    LocalEngine (local GGUF) and GroqEngine (Groq cloud API) at runtime.
    """

    def __init__(self, use_groq: Optional[bool] = None, groq_model: Optional[str] = None):
        self._lock = threading.RLock()
        self._local_engine: Optional[LocalEngine] = None
        self._groq_engine: Optional[GroqEngine] = None
        init_groq = getattr(config, "USE_GROQ", False) if use_groq is None else bool(use_groq)
        self._active_provider = "groq" if init_groq else "local"
        raw = groq_model or getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b")
        self._groq_model = f"openai/{raw}" if raw in ("gpt-oss-20b", "gpt-oss-120b", "gpt-oss-safeguard-20b") else raw

    @property
    def active_provider(self) -> str:
        with self._lock:
            return self._active_provider

    @property
    def is_groq(self) -> bool:
        with self._lock:
            return self._active_provider == "groq"

    @property
    def active_model_name(self) -> str:
        with self._lock:
            if self._active_provider == "groq":
                return getattr(self._groq_engine, "model", getattr(config, "GROQ_MODEL", self._groq_model))
            return os.path.basename(getattr(config, "MODEL_PATH", "model.gguf") or "model.gguf")

    def _get_engine(self, provider: str) -> Union[LocalEngine, GroqEngine]:
        if provider == "groq":
            if self._groq_engine is None:
                current_model = self._groq_model or getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b")
                self._groq_engine = GroqEngine(model_name=current_model)
            return self._groq_engine
        else:
            if self._local_engine is None:
                self._local_engine = LocalEngine()
            return self._local_engine

    def switch(self, provider: str, groq_model: Optional[str] = None) -> None:
        with self._lock:
            if provider not in ("groq", "local"):
                raise ValueError(f"Invalid engine provider: {provider}")
            if groq_model:
                val = groq_model.strip()
                normalized = f"openai/{val}" if val in ("gpt-oss-20b", "gpt-oss-120b", "gpt-oss-safeguard-20b") else val
                self._groq_model = normalized
                config.GROQ_MODEL = normalized
            if provider == "groq":
                target_model = self._groq_model or getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b")
                if self._groq_engine is None or getattr(self._groq_engine, "model", "") != target_model:
                    self._groq_engine = GroqEngine(model_name=target_model)
            elif provider == "local":
                if self._local_engine is None:
                    self._local_engine = LocalEngine()
            config.USE_GROQ = (provider == "groq")
            self._active_provider = provider

    def sync_with_config(self) -> None:
        with self._lock:
            target_provider = "groq" if getattr(config, "USE_GROQ", False) else "local"
            target_model = getattr(config, "GROQ_MODEL", self._groq_model)
            if target_provider != self._active_provider or (target_provider == "groq" and getattr(self._groq_engine, "model", "") != target_model):
                self.switch(target_provider, groq_model=target_model)

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 160,
             temperature: float = 0.7, history: Optional[List[Dict[str, str]]] = None) -> str:
        with self._lock:
            provider = self._active_provider
            try:
                eng = self._get_engine(provider)
            except Exception as eg_err:
                eng = None
                config.log_debug(f"[SwitchableEngine] {provider} init failed: {eg_err}")

        res = ""
        if eng is not None:
            try:
                res = eng.chat(system_prompt, user_prompt, max_tokens=max_tokens,
                               temperature=temperature, history=history)
            except Exception as e:
                config.log_debug(f"[SwitchableEngine] {provider} chat exception: {e}")

        # Fallback 1: if Groq is active and returned nothing or failed, try local
        if (not res or not res.strip()) and provider == "groq":
            print("[SwitchableEngine] Groq returned empty or failed; attempting fallback to local llama_cpp engine...", file=sys.stderr)
            try:
                with self._lock:
                    local_eng = self._get_engine("local")
                res = local_eng.chat(system_prompt, user_prompt, max_tokens=max_tokens,
                                     temperature=temperature, history=history)
            except Exception as e2:
                print(f"[SwitchableEngine] local fallback also failed: {e2}", file=sys.stderr)

        # Fallback 2: if Local is active and returned nothing or failed, try Groq
        if (not res or not res.strip()) and provider == "local" and bool(os.environ.get("GROQ_API_KEY", "").strip()):
            print("[SwitchableEngine] Local engine failed or returned empty; attempting fallback to Groq...", file=sys.stderr)
            try:
                with self._lock:
                    groq_eng = self._get_engine("groq")
                res = groq_eng.chat(system_prompt, user_prompt, max_tokens=max_tokens,
                                    temperature=temperature, history=history)
            except Exception as e3:
                print(f"[SwitchableEngine] Groq fallback also failed: {e3}", file=sys.stderr)

        return res

    def stream_chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 160,
                    temperature: float = 0.7, history: Optional[List[Dict[str, str]]] = None) -> Generator[str, None, None]:
        with self._lock:
            provider = self._active_provider
            try:
                eng = self._get_engine(provider)
            except Exception as eg_err:
                eng = None
                config.log_debug(f"[SwitchableEngine] {provider} stream init failed: {eg_err}")

        yielded_any = False
        if eng is not None:
            try:
                for token in eng.stream_chat(system_prompt, user_prompt, max_tokens=max_tokens,
                                             temperature=temperature, history=history):
                    yielded_any = True
                    yield token
            except Exception as e:
                config.log_debug(f"[SwitchableEngine] {provider} stream exception: {e}")

        # Fallback 1: if Groq stream yielded nothing, try local
        if not yielded_any and provider == "groq":
            print("[SwitchableEngine] Groq stream yielded nothing; attempting fallback to local llama_cpp engine...", file=sys.stderr)
            try:
                with self._lock:
                    local_eng = self._get_engine("local")
                for token in local_eng.stream_chat(system_prompt, user_prompt, max_tokens=max_tokens,
                                                   temperature=temperature, history=history):
                    yield token
            except Exception as e2:
                print(f"[SwitchableEngine] local stream fallback failed: {e2}", file=sys.stderr)

        # Fallback 2: if Local stream yielded nothing, try Groq
        if not yielded_any and provider == "local" and bool(os.environ.get("GROQ_API_KEY", "").strip()):
            print("[SwitchableEngine] Local stream yielded nothing; attempting fallback to Groq...", file=sys.stderr)
            try:
                with self._lock:
                    groq_eng = self._get_engine("groq")
                for token in groq_eng.stream_chat(system_prompt, user_prompt, max_tokens=max_tokens,
                                                  temperature=temperature, history=history):
                    yield token
            except Exception as e3:
                print(f"[SwitchableEngine] Groq stream fallback failed: {e3}", file=sys.stderr)

    def close(self) -> None:
        with self._lock:
            if self._local_engine is not None:
                try:
                    self._local_engine.close()
                except Exception:
                    pass
                self._local_engine = None
            if self._groq_engine is not None:
                try:
                    self._groq_engine.close()
                except Exception:
                    pass
                self._groq_engine = None


def create_switchable_engine(use_groq: Optional[bool] = None, groq_model: Optional[str] = None) -> SwitchableEngine:
    return SwitchableEngine(use_groq=use_groq, groq_model=groq_model)


def create_engine(use_groq: Optional[bool] = None, model_name: Optional[str] = None,
                  switchable: bool = False) -> Union[LocalEngine, GroqEngine, SwitchableEngine]:
    if switchable:
        return create_switchable_engine(use_groq=use_groq, groq_model=model_name)
    should_use_groq = getattr(config, "USE_GROQ", False) if use_groq is None else use_groq
    if should_use_groq:
        model = model_name or getattr(config, "GROQ_MODEL", "gpt-oss-20b")
        return GroqEngine(model_name=model)
    return LocalEngine()


