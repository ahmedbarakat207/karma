"""
LLM engine abstraction. Supports 9Router (DeepSeek V4 Flash), Groq API, and local llama-cpp.
All expose the same .chat() and .stream_chat() interfaces with automatic fallbacks.

stream_chat() yields raw string tokens as the model generates them, enabling
the prosody.py layer to begin TTS synthesis before the full response is ready.
"""
import re
import threading
import requests

import config

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text):
    """Remove <think>...</think> reasoning traces from output."""
    if not text:
        return ""
    # Strip completed thinking tags
    text = _THINK_TAG_RE.sub("", text).strip()
    # Strip unclosed <think> tag if model was cut off during reasoning
    if "<think>" in text:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        else:
            text = text.split("<think>")[0].strip()
    return text


def _strip_thinking_from_stream(token_iter):
    """
    Generator wrapper that filters <think>...</think> blocks out of a token stream.
    Buffers tokens inside a <think> block and discards them.
    """
    buf = ""
    in_think = False
    for token in token_iter:
        buf += token
        while True:
            if not in_think:
                # Look for opening tag
                idx = buf.find("<think>")
                if idx == -1:
                    # No think tag — yield everything except the last 7 chars
                    # (keep a small tail in case the tag spans tokens)
                    safe = buf[:-7] if len(buf) > 7 else ""
                    if safe:
                        yield safe
                        buf = buf[len(safe):]
                    break
                else:
                    # Yield everything before the tag
                    if idx > 0:
                        yield buf[:idx]
                    buf = buf[idx + len("<think>"):]
                    in_think = True
            else:
                # Inside think block — look for closing tag
                idx = buf.find("</think>")
                if idx == -1:
                    # Discard all but the last 9 chars (tail may contain </think>)
                    discard_up_to = max(0, len(buf) - 9)
                    buf = buf[discard_up_to:]
                    break
                else:
                    buf = buf[idx + len("</think>"):]
                    in_think = False
    # Flush remaining buffer (only if not inside a think block)
    if buf and not in_think:
        yield buf


class NineRouterEngine:
    """9Router / OpenRouter DeepSeek V4 Flash engine."""

    def __init__(self):
        self._lock = threading.Lock()
        self.api_base = getattr(config, "NINEROUTER_API_BASE", "http://localhost:20128/v1")
        self.api_key = getattr(config, "NINEROUTER_API_KEY", "sk-1c489e5544334f97-p50fxh-ec2cb811")
        self.model = getattr(config, "NINEROUTER_MODEL", "cmc/deepseek/deepseek-v4-flash")
        print(f"[llm] using 9Router API ({self.model}) at {self.api_base}")

    def chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=12)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return _strip_thinking(text)

    def stream_chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        """Yield raw string tokens as they arrive (SSE streaming)."""
        yield from _strip_thinking_from_stream(
            self._raw_stream(system_prompt, user_prompt, max_tokens, temperature)
        )

    def _raw_stream(self, system_prompt, user_prompt, max_tokens, temperature):
        import json as _json
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        with requests.post(url, headers=headers, json=data, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue


class GroqEngine:
    """Groq-hosted LLM (fast inference, requires API key)."""

    def __init__(self):
        from groq import Groq
        self._lock = threading.Lock()
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.GROQ_MODEL
        print(f"[llm] using Groq API: {self.model}")

    def chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content
        return _strip_thinking(text)

    def stream_chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        """Yield raw string tokens as they arrive via Groq streaming."""
        yield from _strip_thinking_from_stream(
            self._raw_stream(system_prompt, user_prompt, max_tokens, temperature)
        )

    def _raw_stream(self, system_prompt, user_prompt, max_tokens, temperature):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


class LocalEngine:
    """Local llama-cpp model (offline fallback)."""

    def __init__(self):
        from llama_cpp import Llama
        self._lock = threading.Lock()
        self.llm = Llama(
            model_path=config.LOCAL_MODEL_PATH,
            n_ctx=config.LOCAL_CTX_SIZE,
            n_gpu_layers=config.LOCAL_N_GPU_LAYERS,
            verbose=False,
        )
        print(f"[llm] using local model: {config.LOCAL_MODEL_PATH}")

    def chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        with self._lock:
            resp = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        text = resp["choices"][0]["message"]["content"]
        return _strip_thinking(text)

    def stream_chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        """Yield raw string tokens from the local llama-cpp streaming generator."""
        yield from _strip_thinking_from_stream(
            self._raw_stream(system_prompt, user_prompt, max_tokens, temperature)
        )

    def _raw_stream(self, system_prompt, user_prompt, max_tokens, temperature):
        """
        Streams tokens from the local model via a thread+queue pattern.

        Why not `with self._lock: ... yield ...`:
          Holding a lock across a generator yield keeps it locked for the
          entire generation duration (seconds), blocking every other call.
          Instead we run the full generation in a daemon thread that holds
          the lock, and forward tokens out through a queue. The lock is
          released the moment the generation thread finishes, not when the
          last token is consumed by the caller.
        """
        import queue as _queue
        token_q = _queue.Queue()
        _SENTINEL = object()

        def _generate():
            with self._lock:
                try:
                    stream = self.llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
                    for chunk in stream:
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            token_q.put(content)
                except Exception as e:
                    print(f"[llm] local stream error: {e}")
                finally:
                    token_q.put(_SENTINEL)

        t = threading.Thread(target=_generate, daemon=True)
        t.start()
        while True:
            tok = token_q.get()
            if tok is _SENTINEL:
                break
            yield tok


class FallbackEngine:
    """
    Tries primary engine, falls back to secondary on error.

    Error classification:
      - Transient (rate limit 429/413, timeout, 5xx): fall back for this
        call only, then retry primary after TRANSIENT_COOLDOWN_SECONDS.
      - Permanent (auth 401/403, model not found 404): disable primary
        for the session.
    """

    TRANSIENT_COOLDOWN_SECONDS = 30  # retry primary after rate-limit cooldown

    def __init__(self, primary, secondary, primary_name="primary"):
        self._primary = primary
        self._secondary = secondary
        self._primary_name = primary_name
        self._primary_disabled_until = 0   # epoch timestamp; 0 = enabled
        self._primary_permanent_fail = False

    def _primary_available(self):
        if self._primary_permanent_fail:
            return False
        import time as _time
        return _time.time() >= self._primary_disabled_until

    def _handle_error(self, e):
        """Classify error and set appropriate cooldown or permanent disable."""
        import time as _time
        msg = str(e).lower()
        # Transient: rate limits, timeouts, 5xx server errors
        if any(code in msg for code in ("413", "429", "rate_limit", "timeout", "503", "502", "500")):
            print(f"[llm] {self._primary_name} transient error: {e} "
                  f"-- using secondary for {self.TRANSIENT_COOLDOWN_SECONDS}s")
            self._primary_disabled_until = _time.time() + self.TRANSIENT_COOLDOWN_SECONDS
        else:
            # Permanent: auth failure, model not found, etc.
            print(f"[llm] {self._primary_name} permanent failure: {e} -- falling back to secondary permanently")
            self._primary_permanent_fail = True

    def chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        if self._primary_available():
            try:
                return self._primary.chat(system_prompt, user_prompt, max_tokens, temperature)
            except Exception as e:
                self._handle_error(e)
        return self._secondary.chat(system_prompt, user_prompt, max_tokens, temperature)

    def stream_chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        """Stream tokens, falling back to secondary engine on transient errors."""
        if self._primary_available():
            try:
                yield from self._primary.stream_chat(system_prompt, user_prompt, max_tokens, temperature)
                return
            except Exception as e:
                self._handle_error(e)
        yield from self._secondary.stream_chat(system_prompt, user_prompt, max_tokens, temperature)


def create_engine():
    """Factory: returns 9Router (DeepSeek V4 Flash), Groq, or Local fallback."""
    backend = getattr(config, "LLM_BACKEND", "9router")

    if backend == "9router":
        try:
            primary = NineRouterEngine()
            secondary = GroqEngine() if getattr(config, "GROQ_API_KEY", None) else LocalEngine()
            return FallbackEngine(primary, secondary, "9Router (DeepSeek V4 Flash)")
        except Exception as e:
            print(f"[llm] 9Router init failed: {e}")

    if backend == "groq" or getattr(config, "GROQ_API_KEY", None):
        try:
            primary = GroqEngine()
            secondary = LocalEngine()
            return FallbackEngine(primary, secondary, "Groq")
        except Exception as e:
            print(f"[llm] Groq init failed: {e} -- using local model")
            return LocalEngine()

    return LocalEngine()
