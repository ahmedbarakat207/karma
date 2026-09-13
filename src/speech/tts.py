#!/usr/bin/env python3
import io
import os
import re
import sys
import threading
import time
import zipfile
from typing import Optional, List, Dict, Any
import numpy as np
import sounddevice as sd

from src import config
from src.state import internal_state
from src.speech.arabic_g2p import is_arabic, ArabicG2P, EXTRA_SYMBOLS, clean_phonemes


def _ensure_num2words() -> None:
    """Ensure num2words is available in sys.modules so Kokoro/Misaki never crash on import.

    Kokoro and Misaki depend on num2words for English G2P normalization.
    If num2words is not installed in the python environment, provide a built-in
    fallback implementation so TTS never crashes with ModuleNotFoundError.
    """
    try:
        import num2words  # noqa: F401
    except ImportError:
        import types
        import importlib.machinery

        def _fallback_num2words(n, to="cardinal", **kwargs):
            try:
                num = float(n) if "." in str(n) else int(n)
            except Exception:
                return str(n)

            ONES = [
                "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen"
            ]
            TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            ORDINALS = {
                0: "zeroth", 1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
                6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
                11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
                15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
                19: "nineteenth", 20: "twentieth", 30: "thirtieth", 40: "fortieth",
                50: "fiftieth", 60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth"
            }

            def _int_to_cardinal(val: int) -> str:
                if val < 0:
                    return "minus " + _int_to_cardinal(-val)
                if val < 20:
                    return ONES[val]
                if val < 100:
                    rem = val % 10
                    return TENS[val // 10] + (("-" + ONES[rem]) if rem else "")
                if val < 1000:
                    rem = val % 100
                    return ONES[val // 100] + " hundred" + ((" " + _int_to_cardinal(rem)) if rem else "")
                if val < 1000000:
                    thousands = val // 1000
                    rem = val % 1000
                    return _int_to_cardinal(thousands) + " thousand" + ((" " + _int_to_cardinal(rem)) if rem else "")
                if val < 1000000000:
                    millions = val // 1000000
                    rem = val % 1000000
                    return _int_to_cardinal(millions) + " million" + ((" " + _int_to_cardinal(rem)) if rem else "")
                return str(val)

            if isinstance(num, float):
                int_part = int(num)
                dec_part = str(num).split(".")[1]
                dec_words = " ".join(ONES[int(d)] for d in dec_part if d.isdigit())
                return f"{_int_to_cardinal(int_part)} point {dec_words}"

            val = int(num)
            if to == "year" and 1000 <= val <= 2999:
                century = val // 100
                rest = val % 100
                if rest == 0:
                    return f"{_int_to_cardinal(century)} hundred"
                elif rest < 10:
                    return f"{_int_to_cardinal(century)} oh {ONES[rest]}"
                else:
                    return f"{_int_to_cardinal(century)} {_int_to_cardinal(rest)}"

            cardinal = _int_to_cardinal(val)
            if to == "ordinal":
                if val in ORDINALS:
                    return ORDINALS[val]
                if val % 100 in ORDINALS:
                    base = _int_to_cardinal(val - (val % 100))
                    return f"{base} {ORDINALS[val % 100]}"
                if val % 10 in ORDINALS and val % 10 != 0:
                    base = _int_to_cardinal(val - (val % 10))
                    return f"{base}-{ORDINALS[val % 10]}"
                return cardinal + "th"

            return cardinal

        mod = types.ModuleType("num2words")
        mod.num2words = _fallback_num2words
        mod.__spec__ = importlib.machinery.ModuleSpec("num2words", None)
        mod.__file__ = "<fallback_num2words>"
        sys.modules["num2words"] = mod


_ensure_num2words()


def clean_for_speech(text: str) -> str:
    """Clean text for speech synthesis while preserving English and Arabic script and punctuation."""
    if not text:
        return ""

    text = re.sub(r"\*.*?\*", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    # Preserve alphanumeric, English, Arabic unicode range, diacritics, and both English/Arabic punctuation
    text = re.sub(r"[^\w\s.,!?'\-،؟\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def set_system_volume_max() -> None:
    """Automatically unmute and set ALSA, PulseAudio, and PipeWire volume to 100% across all cards and controls."""
    if not sys.platform.startswith("linux"):
        return
    import shutil
    import subprocess

    # 1. ALSA mixer controls
    if shutil.which("amixer"):
        controls = ["Headphone", "Headphones", "Master", "PCM", "Speaker", "Playback"]
        cards = ["Headphones", "0", "1", "2", "3", "Device"]
        for ctrl in controls:
            try:
                subprocess.run(["amixer", "sset", ctrl, "100%", "unmute"], capture_output=True, timeout=1)
            except Exception:
                pass
            for card in cards:
                try:
                    subprocess.run(["amixer", "-c", str(card), "sset", ctrl, "100%", "unmute"], capture_output=True, timeout=1)
                except Exception:
                    pass
        # Force Raspberry Pi bcm2835 audio routing to 3.5mm analog jack (numid=3 1)
        try:
            subprocess.run(["amixer", "cset", "numid=3", "1"], capture_output=True, timeout=1)
            subprocess.run(["amixer", "-c", "Headphones", "cset", "numid=3", "1"], capture_output=True, timeout=1)
        except Exception:
            pass

    # 2. PulseAudio controls (if PulseAudio daemon is running)
    if shutil.which("pactl"):
        try:
            res = subprocess.run(["pactl", "list", "sinks", "short"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 2:
                        sname = parts[1]
                        if "hdmi" in sname.lower():
                            subprocess.run(["pactl", "suspend-sink", sname, "1"], capture_output=True, timeout=1)
                            subprocess.run(["pactl", "set-sink-mute", sname, "1"], capture_output=True, timeout=1)
                        elif any(k in sname.lower() for k in ("usb", "uac", "dac", "speaker", "headphone", "analog", "bcm2835")):
                            subprocess.run(["pactl", "suspend-sink", sname, "0"], capture_output=True, timeout=1)
                            subprocess.run(["pactl", "set-default-sink", sname], capture_output=True, timeout=1)
                            subprocess.run(["pactl", "set-sink-mute", sname, "0"], capture_output=True, timeout=1)
                            subprocess.run(["pactl", "set-sink-volume", sname, "100%"], capture_output=True, timeout=1)
        except Exception:
            pass

    # 3. PipeWire controls (if PipeWire is running)
    if shutil.which("wpctl"):
        try:
            subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"], capture_output=True, timeout=1)
            subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "1.0"], capture_output=True, timeout=1)
        except Exception:
            pass




def _normalize_audio(audio: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if audio is None or len(audio) == 0:
        return audio

    max_val = float(np.max(np.abs(audio)))
    if max_val > 0.01:
        audio = (audio / max_val) * 0.98

    fade_len = min(60, len(audio) // 4)
    if fade_len > 4:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
        audio[:fade_len] *= fade_in
        audio[-fade_len:] *= fade_out

    return np.ascontiguousarray(audio, dtype=np.float32)


class TTSEngine:
    """Bilingual Text-to-Speech engine supporting Kokoro-82M (English) and Nabra-82M (Arabic)

    Automatically routes sentences to the correct neural pipeline based on language detection.
    """

    def __init__(self, lang_code: str = config.TTS_LANG_CODE, voice: str = config.TTS_VOICE,
                 speaking_event: Optional[threading.Event] = None,
                 interrupt_event: Optional[threading.Event] = None):
        self.voice = voice
        self.lang_code = lang_code
        self.speaking_event = speaking_event
        self.interrupt_event = interrupt_event
        self._synth_lock = threading.Lock()
        self.onnx_session = None
        self.onnx_kokoro = None
        self._voices_cache: Dict[str, np.ndarray] = {}

        # English pipeline (Kokoro-82M PyTorch) — lazy: only built if ONNX
        # fast path is unavailable or fails. Building KPipeline pulls torch
        # weights and runs a slow warmup, so never do it eagerly here.
        self.en_pipeline = None
        self._en_init_attempted = False
        self.pipeline = None  # for backward compatibility; set on lazy init

        # Arabic pipeline (Nabra-82M) - initialized lazily on first Arabic request
        self.ar_pipeline = None
        self.ar_voice = None
        self.ar_g2p = None
        self._ar_init_attempted = False

        model_path = getattr(config, "KOKORO_MODEL_PATH", "")
        voices_path = getattr(config, "KOKORO_VOICES_PATH", "")

        # Auto-fetch the quantized ONNX voice files from their correct
        # upstream locations when USE_KOKORO_ONNX is on but files are absent.
        # (setup.sh previously pointed at hexgrad/Kokoro-82M filenames that
        # don't exist there, so Pi installs silently fell back to slow torch.)
        if getattr(config, "USE_KOKORO_ONNX", False):
            try:
                model_path, voices_path = self._ensure_onnx_files(model_path, voices_path)
            except Exception as e:
                config.log_debug(f"[speech] ONNX auto-download note: {e}")

        if getattr(config, "USE_KOKORO_ONNX", False) and model_path and os.path.exists(model_path) and voices_path and os.path.exists(voices_path):
            try:
                import onnxruntime as ort
                from kokoro_onnx import Kokoro
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = getattr(config, "TTS_THREADS", getattr(config, "N_THREADS", 4))
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
                self.onnx_session = session
                try:
                    self.onnx_kokoro = Kokoro.from_session(session, voices_path)
                except Exception:
                    # Older kokoro-onnx without from_session: construct directly
                    # (builds a second session; still far faster than torch).
                    self.onnx_kokoro = Kokoro(model_path, voices_path)
                print(f"[speech] ONNX Kokoro ready ({os.path.basename(model_path)}, voice={self.voice})")
            except Exception as e:
                print(f"[speech] ONNX init note: {e}", file=sys.stderr)
                config.log_debug(f"[speech] ONNX init note: {e}")
                self.onnx_session = None
                self.onnx_kokoro = None
        elif getattr(config, "USE_KOKORO_ONNX", False):
            print(f"[speech] ONNX enabled but files missing ({model_path}, {voices_path}); using PyTorch fallback (slow).", file=sys.stderr)

        try:
            threading.Thread(target=set_system_volume_max, daemon=True, name="auto_max_volume").start()
        except Exception:
            pass

        # No blocking warmup here: synthesis of even "warmup" costs 10-40s
        # of torch/ONNX inference on Pi CPUs and stalls the whole robot
        # startup (watchdog restarts). First real utterance warms the model.

    def _ensure_onnx_files(self, model_path: str, voices_path: str):
        """Download missing ONNX model/voices from correct upstreams. Returns (model_path, voices_path)."""
        import shutil
        import subprocess
        import urllib.request

        models_dir = getattr(config, "MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models"))
        models_dir = os.path.abspath(models_dir)
        os.makedirs(models_dir, exist_ok=True)

        if not model_path:
            model_path = os.path.join(models_dir, "kokoro_q4.onnx")
        if not voices_path:
            voices_path = os.path.join(models_dir, "voices-v1.0.bin")

        if not os.path.exists(model_path):
            try:
                from huggingface_hub import hf_hub_download
                print(f"[speech] downloading Kokoro ONNX model (onnx-community/Kokoro-82M-v1.0-ONNX:onnx/model_q4.onnx)...")
                dl = hf_hub_download(repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", filename="onnx/model_q4.onnx", local_dir=models_dir)
                # hf_hub_download places it under models_dir/onnx/; move to expected path
                if os.path.abspath(dl) != os.path.abspath(model_path):
                    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
                    shutil.move(dl, model_path)
                print(f"[speech] ONNX model ready: {model_path}")
            except Exception as e:
                config.log_debug(f"[speech] ONNX model download note: {e}")

        if not os.path.exists(voices_path):
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
            try:
                print(f"[speech] downloading Kokoro voices (~27MB)...")
                os.makedirs(os.path.dirname(os.path.abspath(voices_path)), exist_ok=True)
                tmp = voices_path + ".part"
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                os.replace(tmp, voices_path)
                print(f"[speech] voices ready: {voices_path}")
            except Exception as e:
                # curl fallback (better retry/proxy handling on Pi)
                try:
                    if shutil.which("curl"):
                        subprocess.run(["curl", "-L", "--retry", "3", "-o", voices_path, url], check=True, timeout=300)
                    else:
                        raise e
                except Exception as e2:
                    config.log_debug(f"[speech] voices download note: {e2}")
        return model_path, voices_path

    def _init_english_torch(self):
        """Lazy PyTorch KPipeline init (slow). Only used when ONNX is unavailable."""
        if self.en_pipeline is not None:
            return self.en_pipeline
        if self._en_init_attempted:
            return None
        self._en_init_attempted = True
        _ensure_num2words()
        try:
            from kokoro import KPipeline
            self.en_pipeline = KPipeline(lang_code=self.lang_code)
            self.pipeline = self.en_pipeline
            return self.en_pipeline
        except Exception as e:
            config.log_debug(f"[speech] Kokoro English pipeline init error: {e}")
            self.en_pipeline = None
            return None

    def _init_nabra(self) -> Optional[Any]:
        """Lazy-initialize Nabra-82M Arabic TTS pipeline."""
        if self.ar_pipeline is not None:
            return self.ar_pipeline
        if self._ar_init_attempted:
            return None

        self._ar_init_attempted = True
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from kokoro import KModel, KPipeline
            from kokoro import pipeline as kpipeline_mod

            repo_id = getattr(config, "NABRA_REPO_ID", "oddadmix/Nabra-82M-v0.1")
            nabra_dir = getattr(config, "NABRA_MODEL_DIR", os.path.join(config.MODELS_DIR, "nabra"))
            os.makedirs(nabra_dir, exist_ok=True)

            cfg_file = os.path.join(nabra_dir, "config.json")
            model_file = os.path.join(nabra_dir, "kokoro_arabic.pth")
            voice_file = os.path.join(nabra_dir, "af_msa.pt")

            if not os.path.exists(cfg_file):
                config.log_debug(f"[speech] downloading Nabra config from {repo_id}...")
                cfg_file = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=nabra_dir)
            if not os.path.exists(model_file):
                config.log_debug(f"[speech] downloading Nabra model weights from {repo_id}...")
                model_file = hf_hub_download(repo_id=repo_id, filename="kokoro_arabic.pth", local_dir=nabra_dir)
            if not os.path.exists(voice_file):
                config.log_debug(f"[speech] downloading Nabra voice from {repo_id}...")
                voice_file = hf_hub_download(repo_id=repo_id, filename="af_msa.pt", local_dir=nabra_dir)

            kmodel = KModel(repo_id=repo_id, config=cfg_file, model=model_file, disable_complex=True).eval()
            kmodel.vocab.update(EXTRA_SYMBOLS)

            kpipeline_mod.LANG_CODES.setdefault("ar", "ar")
            pipeline = KPipeline(lang_code="ar", repo_id=repo_id, model=kmodel)
            _orig_g2p = pipeline.g2p

            def _nabra_g2p(t):
                ph, toks = _orig_g2p(t)
                return clean_phonemes(ph), toks

            pipeline.g2p = _nabra_g2p

            self.ar_voice = torch.load(voice_file, map_location="cpu", weights_only=True)
            self.ar_pipeline = pipeline
            self.ar_g2p = ArabicG2P(diacritize=False)
            config.log_debug("[speech] Nabra-82M Arabic TTS pipeline ready!")
            return self.ar_pipeline
        except Exception as e:
            config.log_debug(f"[speech] Nabra Arabic TTS init note: {e}")
            return None

    def _load_voice_tensor(self, voices_path: str, voice_name: str) -> Optional[np.ndarray]:
        if voice_name in self._voices_cache:
            return self._voices_cache[voice_name]
        try:
            with zipfile.ZipFile(voices_path, "r") as z:
                target_file = f"{voice_name}.npy"
                if target_file in z.namelist():
                    with z.open(target_file) as f:
                        arr = np.load(io.BytesIO(f.read()))
                        self._voices_cache[voice_name] = arr
                        return arr
        except Exception as e:
            config.log_debug(f"[speech] voice load error: {e}")
        return None

    def _synthesize_arabic(self, text: str, speed: float = 1.0) -> Optional[np.ndarray]:
        """Synthesize Arabic text using Nabra-82M."""
        pipeline = self._init_nabra()
        if pipeline is None or self.ar_voice is None:
            config.log_debug("[speech] Nabra-82M pipeline not ready, cannot synthesize Arabic.")
            return None

        try:
            chunks: List[np.ndarray] = []
            for _, _, audio in pipeline(text, voice=self.ar_voice, speed=speed):
                if audio is not None:
                    if hasattr(audio, "detach"):
                        arr = audio.detach().cpu().numpy().flatten().astype(np.float32)
                    else:
                        arr = np.asarray(audio, dtype=np.float32).flatten()
                    if len(arr) > 0:
                        chunks.append(arr)
            if chunks:
                full = np.concatenate(chunks).astype(np.float32)
                return _normalize_audio(full)
        except Exception as e:
            config.log_debug(f"[speech] Nabra Arabic synthesis error: {e}")
        return None

    def _synthesize_english(self, text: str, speed: float = 1.0) -> Optional[np.ndarray]:
        """Synthesize English text using Kokoro-82M (ONNX fast path, PyTorch fallback)."""
        # Fast path: quantized ONNX via kokoro-onnx (no torch, ~3-10x faster on Pi).
        if self.onnx_kokoro is not None:
            try:
                import time
                t0 = time.time()
                audio, sr = self.onnx_kokoro.create(text, voice=self.voice, speed=float(speed), lang="en-us")
                if audio is not None and len(audio) > 0:
                    arr = np.asarray(audio, dtype=np.float32).flatten()
                    # kokoro-onnx outputs 24kHz; resample only if engine rate differs
                    if sr and int(sr) != int(config.TTS_SAMPLE_RATE):
                        try:
                            import scipy.signal
                            arr = scipy.signal.resample_poly(
                                arr, int(config.TTS_SAMPLE_RATE), int(sr)).astype(np.float32)
                        except Exception:
                            pass
                    config.log_debug(f"[speech] ONNX synth {len(arr)} samples in {time.time()-t0:.1f}s")
                    return _normalize_audio(arr)
            except Exception as e:
                print(f"[speech] ONNX synthesis note: {e}", file=sys.stderr)
                config.log_debug(f"[speech] ONNX synthesis note: {e}")
                # fall through to PyTorch

        # Legacy manual ONNX session path (kept if onnx_kokoro missing but
        # raw session exists). Uses model.vocab (kokoro>=0.9 moved it off KPipeline).
        if self.onnx_kokoro is None and self.onnx_session is not None:
            try:
                pipe = self._init_english_torch()
                model = getattr(pipe, "model", None) if pipe is not None else None
                vocab = getattr(model, "vocab", None) if model is not None else None
                voices_path = getattr(config, "KOKORO_VOICES_PATH", "")
                voice_arr = self._load_voice_tensor(voices_path, self.voice)
                if pipe is not None and vocab and voice_arr is not None:
                    ps, _ = pipe.g2p(text)
                    if ps:
                        ids = [vocab[c] for c in ps if c in vocab]
                        if ids:
                            import numpy as _np
                            input_ids = _np.array([[0] + ids + [0]], dtype=_np.int64)
                            tokens_len = len(input_ids[0])
                            style = _np.ascontiguousarray(voice_arr[min(tokens_len, len(voice_arr) - 1)], dtype=_np.float32)
                            if style.ndim == 1:
                                style = style[_np.newaxis, :]
                            names = {i.name for i in self.onnx_session.get_inputs()}
                            feed = {}
                            feed["input_ids" if "input_ids" in names else "tokens"] = input_ids
                            feed["style"] = style.astype(_np.float32)
                            feed["speed"] = _np.array([float(speed)], dtype=_np.float32)
                            waveform = self.onnx_session.run(None, feed)[0]
                            if waveform is not None:
                                return _normalize_audio(_np.asarray(waveform).flatten().astype(_np.float32))
            except Exception as e:
                config.log_debug(f"[speech] manual ONNX note: {e}")

        # Slow path: PyTorch KPipeline (lazy init so startup never blocks).
        try:
            pipe = self._init_english_torch()
            if pipe is None:
                config.log_debug("[speech] English pipeline not available.")
                return None
            chunks: List[np.ndarray] = []
            for _, _, audio in pipe(text, voice=self.voice, speed=speed):
                if audio is not None:
                    if hasattr(audio, "detach"):
                        arr = audio.detach().cpu().numpy().flatten().astype(np.float32)
                    else:
                        arr = np.asarray(audio, dtype=np.float32).flatten()
                    if len(arr) > 0:
                        chunks.append(arr)
            if chunks:
                full = np.concatenate(chunks).astype(np.float32)
                return _normalize_audio(full)
        except Exception as e:
            print(f"[speech] English synthesis error: {e}", file=sys.stderr)
            config.log_debug(f"[speech] English synthesis error: {e}")
        return None

    def _synthesize(self, text: str, speed: float = 1.0) -> Optional[np.ndarray]:
        spoken = clean_for_speech(text)
        if not spoken:
            return None

        with self._synth_lock:
            try:
                # Serialize against LLM inference: overlapping onnxruntime and
                # llama.cpp thread pools browns out marginal Pi PSUs (SIGSEGV).
                # Playback stays concurrent — only synthesis takes the gate.
                from src.cognition.engine import heavy_compute
                gate = heavy_compute()
            except Exception:
                import contextlib
                gate = contextlib.nullcontext()
            with gate:
                try:
                    use_arabic = is_arabic(spoken) and getattr(config, "NABRA_ENABLED", True)
                    if use_arabic:
                        audio = self._synthesize_arabic(spoken, speed=speed)
                        if audio is not None:
                            return audio
                        # Fallback to English pipeline if Arabic synthesis failed
                        return self._synthesize_english(spoken, speed=speed)
                    else:
                        return self._synthesize_english(spoken, speed=speed)
                except Exception as e:
                    config.log_debug(f"[speech] synthesis error: {e}")
                    return None

    def _find_best_pulse_sink(self) -> Optional[str]:
        """Find non-HDMI PulseAudio/PipeWire sink."""
        if not sys.platform.startswith("linux"):
            return None
        import shutil
        import subprocess
        if not shutil.which("pactl"):
            return None
        try:
            res = subprocess.run(["pactl", "list", "sinks", "short"], capture_output=True, text=True, timeout=2)
            if res.returncode != 0 or not res.stdout:
                return None
            sinks = []
            for line in res.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    sinks.append(parts[1])
            # 1. USB Audio
            for s in sinks:
                if any(k in s.lower() for k in ("usb", "uac", "dac", "speaker", "codec")):
                    return s
            # 2. 3.5mm analog headphones
            for s in sinks:
                if any(k in s.lower() for k in ("headphone", "analog", "bcm2835")) and "hdmi" not in s.lower():
                    return s
            # 3. Any other non-HDMI sink
            for s in sinks:
                if "hdmi" not in s.lower():
                    return s
            return None
        except Exception:
            return None

    def _find_best_alsa_devices(self) -> List[str]:
        """Find non-HDMI ALSA playback devices on Linux/Raspberry Pi.

        Routing audio to HDMI (vc4-hdmi) on Raspberry Pi causes the HDMI clock
        to re-synchronize, which blanks the 7-inch LCD display (turns off and on again)
        and sends audio to a display that has no speakers.
        """
        configured = (getattr(config, "AUDIO_OUTPUT_DEVICE", "") or os.environ.get("AUDIO_OUTPUT_DEVICE", "")).strip()
        if configured:
            return [configured]

        if not sys.platform.startswith("linux"):
            return []

        import shutil
        if not shutil.which("aplay"):
            return []

        try:
            import subprocess
            proc = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=2)
            if proc.returncode != 0 or not proc.stdout:
                return []

            cards = []
            for line in proc.stdout.splitlines():
                m = re.match(r"^card\s+(\d+):\s+([\w\-]+)\s+\[([^\]]+)\]", line)
                if m:
                    card_id, card_name, card_desc = m.group(1), m.group(2), m.group(3)
                    cards.append((card_id, card_name, card_desc))

            candidates = []

            # Priority 1: USB audio devices (headphones, dongles, speakers, DACs)
            for cid, cname, cdesc in cards:
                text = (cname + " " + cdesc).lower()
                if any(k in text for k in ("usb", "uac", "dac", "speaker", "codec")):
                    for dev in (f"plughw:CARD={cname},DEV=0", f"plughw:{cid},0", f"sysdefault:CARD={cname}", f"sysdefault:{cid}", f"hw:{cid},0"):
                        if dev not in candidates:
                            candidates.append(dev)

            # Priority 2: 3.5mm analog headphone jack (bcm2835 Headphones)
            for cid, cname, cdesc in cards:
                text = (cname + " " + cdesc).lower()
                if any(k in text for k in ("headphone", "analog", "bcm2835")) and "hdmi" not in text:
                    for dev in (f"plughw:CARD={cname},DEV=0", f"plughw:{cid},0", f"sysdefault:CARD={cname}", f"sysdefault:{cid}", f"hw:{cid},0"):
                        if dev not in candidates:
                            candidates.append(dev)

            # Priority 3: Any non-HDMI card
            for cid, cname, cdesc in cards:
                text = (cname + " " + cdesc).lower()
                if "hdmi" not in text:
                    for dev in (f"plughw:CARD={cname},DEV=0", f"plughw:{cid},0", f"sysdefault:CARD={cname}", f"sysdefault:{cid}", f"hw:{cid},0"):
                        if dev not in candidates:
                            candidates.append(dev)

            # Priority 4: PulseAudio ALSA plugin ONLY if an active non-HDMI sink exists
            pulse_sink = self._find_best_pulse_sink()
            if pulse_sink and "pulse" not in candidates:
                candidates.append("pulse")

            return candidates
        except Exception as e:
            print(f"[speech] device discovery error: {e}", file=sys.stderr)
            return []

    def _play_audio(self, audio: Optional[np.ndarray]) -> bool:
        if audio is None or len(audio) == 0:
            return False
        if self.interrupt_event and self.interrupt_event.is_set():
            return False

        played = False
        backend_used = None
        errors: List[str] = []

        try:
            audio_arr = np.ascontiguousarray(audio, dtype=np.float32)
            internal_state.set_playing_audio(True)
            try:
                from src.ui.server import broadcast_state_threadsafe
                broadcast_state_threadsafe()
            except Exception:
                pass

            if not getattr(self, "_volume_maxed", False):
                self._volume_maxed = True
                set_system_volume_max()

            # Native 24kHz mono first: aplay/paplay go through the ALSA plug
            # layer which resamples in C far faster than scipy in Python on
            # Pi CPUs. Only pay for the scipy 44.1kHz stereo resample if the
            # native file fails on every hardware backend.
            int16_mono = np.clip(audio_arr * 32767.0, -32768, 32767).astype(np.int16)
            resampled = None

            def _write_wav(path: str, pcm: np.ndarray, channels: int, rate: int) -> None:
                import wave as _wave
                with _wave.open(path, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(2)
                    wf.setframerate(rate)
                    if channels == 2:
                        wf.writeframes(np.column_stack((pcm, pcm)).flatten().tobytes())
                    else:
                        wf.writeframes(pcm.tobytes())

            def _ensure_resampled() -> tuple:
                nonlocal resampled
                if resampled is not None:
                    return resampled
                try:
                    import scipy.signal
                    rs = scipy.signal.resample_poly(audio_arr, 147, 80).astype(np.float32)
                    rs16 = np.clip(rs * 32767.0, -32768, 32767).astype(np.int16)
                    resampled = (rs, rs16)
                except Exception:
                    resampled = (audio_arr, int16_mono)
                return resampled

            import tempfile
            import wave
            import shutil
            import subprocess

            tmp_wav = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tmp_wav = tf.name

                _write_wav(tmp_wav, int16_mono, 1, int(config.TTS_SAMPLE_RATE))

                # Method 1: PulseAudio native paplay explicitly targeted to verified NON-HDMI sink
                pulse_sink = self._find_best_pulse_sink()
                if not played and pulse_sink and shutil.which("paplay"):
                    try:
                        p_res = subprocess.run(["paplay", "--device", pulse_sink, tmp_wav], capture_output=True, text=True, timeout=30)
                        if p_res.returncode == 0:
                            played = True
                            backend_used = f"paplay ({pulse_sink})"
                        else:
                            errors.append(f"paplay {pulse_sink}: {p_res.stderr.strip()[:80]}")
                    except Exception as pe:
                        errors.append(f"paplay: {pe}")

                # Method 2: ALSA aplay on prioritized non-HDMI hardware devices
                # (native 24kHz mono — plug layer resamples if needed).
                if not played and shutil.which("aplay"):
                    devices_to_try = self._find_best_alsa_devices()
                    if not devices_to_try and getattr(config, "ALLOW_HDMI_AUDIO", False):
                        devices_to_try = ["default"]

                    for dev in devices_to_try:
                        try:
                            cmd = ["aplay", "-D", dev, tmp_wav] if dev != "default" else ["aplay", tmp_wav]
                            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                            if proc.returncode == 0:
                                played = True
                                backend_used = f"aplay ({dev})"
                                break
                            else:
                                err_text = proc.stderr.strip().replace("\n", " ")
                                errors.append(f"aplay {dev}: {err_text[:80]}")
                        except Exception as e:
                            errors.append(f"aplay {dev}: {e}")

                # Only now pay for 44.1kHz stereo (some DACs reject 24k mono).
                if not played:
                    _, int16_resampled = _ensure_resampled()
                    _write_wav(tmp_wav, int16_resampled, 2, 44100)

                # Method 3: sounddevice strictly targeted to non-HDMI output index
                if not played:
                    try:
                        import sounddevice as sd
                        sd_dev = None
                        all_devs = sd.query_devices()
                        # Prefer USB DAC
                        for idx, d in enumerate(all_devs):
                            if d.get("max_output_channels", 0) > 0:
                                n = d.get("name", "").lower()
                                if any(k in n for k in ("usb", "uac", "dac", "speaker", "codec")):
                                    sd_dev = idx
                                    break
                        # Then prefer Headphones / bcm2835 (non-HDMI)
                        if sd_dev is None:
                            for idx, d in enumerate(all_devs):
                                if d.get("max_output_channels", 0) > 0:
                                    n = d.get("name", "").lower()
                                    if any(k in n for k in ("headphone", "analog", "bcm2835")) and "hdmi" not in n:
                                        sd_dev = idx
                                        break
                        # On Linux, ONLY play if an explicit non-HDMI device index was found!
                        if sd_dev is not None or not sys.platform.startswith("linux"):
                            rs_float, _ = _ensure_resampled()
                            play_data = rs_float
                            rate = 44100
                            sd.play(play_data, rate, device=sd_dev)
                            sd.wait()
                            played = True
                            backend_used = f"sounddevice (device #{sd_dev})"
                    except Exception as sde:
                        errors.append(f"sounddevice: {sde}")


                # Method 4: ffplay / mpv fallback
                if not played:
                    for player, player_cmd in [("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_wav]),
                                               ("mpv", ["mpv", "--no-video", tmp_wav])]:
                        if shutil.which(player):
                            try:
                                proc = subprocess.run(player_cmd, capture_output=True, timeout=30)
                                if proc.returncode == 0:
                                    played = True
                                    backend_used = player
                                    break
                            except Exception:
                                pass

            finally:
                if tmp_wav and os.path.exists(tmp_wav):
                    try:
                        os.remove(tmp_wav)
                    except Exception:
                        pass

            if played:
                print(f"[speech] 🔊 audio played via {backend_used}")
            else:
                print(f"[speech] ⚠️ all audio playback attempts failed. Errors: {'; '.join(errors)}", file=sys.stderr)
                # Visual fallback: keep mouth animation running for the duration of the speech
                duration = max(1.5, len(audio_arr) / config.TTS_SAMPLE_RATE)
                t0 = time.time()
                while time.time() - t0 < duration:
                    if self.interrupt_event and self.interrupt_event.is_set():
                        break
                    time.sleep(0.05)
        except Exception as e:
            print(f"[speech] playback exception: {e}", file=sys.stderr)
        finally:
            internal_state.set_playing_audio(False)
            try:
                from src.ui.server import broadcast_state_threadsafe
                broadcast_state_threadsafe()
            except Exception:
                pass
        return played

    def speak(self, text: str, speed: float = 1.0) -> bool:
        spoken_text = clean_for_speech(text)
        if not spoken_text:
            return False

        if self.interrupt_event:
            self.interrupt_event.clear()
        if self.speaking_event:
            self.speaking_event.set()

        played = False
        try:
            audio = self._synthesize(spoken_text, speed=speed)
            if audio is None:
                print(f"[speech] ⚠️ TTS synthesis returned no audio for: '{spoken_text[:40]}...'", file=sys.stderr)
                return False
            print(f"[speech] synthesized {len(audio)} samples ({len(audio)/config.TTS_SAMPLE_RATE:.1f}s), playing...")
            if not (self.interrupt_event and self.interrupt_event.is_set()):
                played = self._play_audio(audio)
        except Exception as e:
            print(f"[speech] TTS speak error: {e}", file=sys.stderr)
        finally:
            if self.speaking_event:
                self.speaking_event.clear()
        return played

