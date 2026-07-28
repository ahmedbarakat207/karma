"""
LLM engine abstraction. Supports 9Router (DeepSeek V4 Flash), Groq API, and local llama-cpp.
All expose the same .chat() interface with automatic fallbacks.
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


class NineRouterEngine:
    """9Router / OpenRouter DeepSeek V4 Flash engine."""

    def __init__(self):
        self._lock = threading.Lock()
        self.api_base = getattr(config, "NINEROUTER_API_BASE", "http://localhost:20128/v1")
        self.api_key = getattr(config, "NINEROUTER_API_KEY", "sk-1c489e5544334f97-p50fxh-ec2cb811")
        self.model = getattr(config, "NINEROUTER_MODEL", "cmc/deepseek/deepseek-v4-flash")
        print(f"[llm] using 9Router API ({self.model}) at {self.api_base}")

    def chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        with self._lock:
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


class GroqEngine:
    """Groq-hosted LLM (fast inference, requires API key)."""

    def __init__(self):
        from groq import Groq
        self._lock = threading.Lock()
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.GROQ_MODEL
        print(f"[llm] using Groq API: {self.model}")

    def chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        with self._lock:
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


class FallbackEngine:
    """Tries primary engine, falls back to secondary on error."""

    def __init__(self, primary, secondary, primary_name="primary"):
        self._primary = primary
        self._secondary = secondary
        self._primary_name = primary_name
        self._use_primary = True

    def chat(self, system_prompt, user_prompt, max_tokens=300, temperature=0.7):
        if self._use_primary:
            try:
                return self._primary.chat(system_prompt, user_prompt, max_tokens, temperature)
            except Exception as e:
                print(f"[llm] {self._primary_name} failed: {e} -- falling back to secondary")
                self._use_primary = False
        return self._secondary.chat(system_prompt, user_prompt, max_tokens, temperature)


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
