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


def _normalize_audio(audio: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if audio is None or len(audio) == 0:
        return audio

    max_val = float(np.max(np.abs(audio)))
    if max_val > 0.95:
        audio = (audio / max_val) * 0.88
    elif 0.05 < max_val < 0.60:
        audio = (audio / max_val) * 0.85

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
        self.speaking_event = speaking_event
        self.interrupt_event = interrupt_event
        self._synth_lock = threading.Lock()
        self.onnx_session = None
        self._voices_cache: Dict[str, np.ndarray] = {}

        # English pipeline (Kokoro-82M)
        _ensure_num2words()
        try:
            from kokoro import KPipeline
            self.en_pipeline = KPipeline(lang_code=lang_code)
        except Exception as e:
            config.log_debug(f"[speech] Kokoro English pipeline init error: {e}")
            self.en_pipeline = None
        self.pipeline = self.en_pipeline  # for backward compatibility

        # Arabic pipeline (Nabra-82M) - initialized lazily on first Arabic request
        self.ar_pipeline = None
        self.ar_voice = None
        self.ar_g2p = None
        self._ar_init_attempted = False

        model_path = getattr(config, "KOKORO_MODEL_PATH", "")
        voices_path = getattr(config, "KOKORO_VOICES_PATH", "")

        if getattr(config, "USE_KOKORO_ONNX", False) and model_path and os.path.exists(model_path) and voices_path and os.path.exists(voices_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = getattr(config, "N_THREADS", 4)
                opts.inter_op_num_threads = 1
                self.onnx_session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
                self._load_voice_tensor(voices_path, self.voice)
                config.log_debug(f"[speech] initialized Quantized Kokoro-82M ONNX model from {model_path}!")
            except Exception as e:
                config.log_debug(f"[speech] ONNX init note: {e}")
                self.onnx_session = None

        try:
            self._synthesize("warmup", speed=1.0)
        except Exception:
            pass

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
            pipeline.g2p = lambda t: (clean_phonemes(_orig_g2p(t)[0]), _orig_g2p(t)[1])

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
        """Synthesize English text using Kokoro-82M (ONNX or PyTorch)."""
        if self.en_pipeline is None:
            config.log_debug("[speech] English pipeline not available.")
            return None
        try:
            if self.onnx_session is not None:
                voices_path = getattr(config, "KOKORO_VOICES_PATH", "")
                voice_arr = self._load_voice_tensor(voices_path, self.voice)
                if voice_arr is not None:
                    ps, _ = self.en_pipeline.g2p(text)
                    if ps:
                        input_ids = np.array([[0] + [self.en_pipeline.vocab[c] for c in ps if c in self.en_pipeline.vocab] + [0]], dtype=np.int64)
                        tokens_len = len(input_ids[0])
                        style = np.ascontiguousarray(voice_arr[min(tokens_len, len(voice_arr) - 1)], dtype=np.float32)
                        speed_arr = np.array([float(speed)], dtype=np.float32)
                        waveform = self.onnx_session.run(None, {
                            "input_ids": input_ids,
                            "style": style,
                            "speed": speed_arr
                        })[0]
                        if waveform is not None:
                            return _normalize_audio(waveform.flatten().astype(np.float32))

            chunks: List[np.ndarray] = []
            for _, _, audio in self.en_pipeline(text, voice=self.voice, speed=speed):
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
            config.log_debug(f"[speech] English synthesis error: {e}")
        return None

    def _synthesize(self, text: str, speed: float = 1.0) -> Optional[np.ndarray]:
        spoken = clean_for_speech(text)
        if not spoken:
            return None

        with self._synth_lock:
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
            # Priority 1: USB audio devices (headphones, dongles, speakers)
            for cid, cname, cdesc in cards:
                text = (cname + " " + cdesc).lower()
                if any(k in text for k in ("usb", "uac", "dac", "speaker", "codec")):
                    dev = f"plughw:CARD={cname},DEV=0"
                    if dev not in candidates:
                        candidates.append(dev)

            # Priority 2: 3.5mm analog headphone jack (bcm2835 Headphones)
            for cid, cname, cdesc in cards:
                text = (cname + " " + cdesc).lower()
                if any(k in text for k in ("headphone", "analog")) and "hdmi" not in text:
                    dev = f"plughw:CARD={cname},DEV=0"
                    if dev not in candidates:
                        candidates.append(dev)

            # Priority 3: Any non-HDMI card
            for cid, cname, cdesc in cards:
                text = (cname + " " + cdesc).lower()
                if "hdmi" not in text:
                    dev = f"plughw:CARD={cname},DEV=0"
                    if dev not in candidates:
                        candidates.append(dev)

            return candidates
        except Exception as e:
            config.log_debug(f"[speech] device discovery error: {e}")
            return []

    def _play_audio(self, audio: Optional[np.ndarray]) -> bool:
        if audio is None or len(audio) == 0:
            return False
        if self.interrupt_event and self.interrupt_event.is_set():
            return False

        played = False
        try:
            audio_arr = np.ascontiguousarray(audio, dtype=np.float32)
            internal_state.set_playing_audio(True)
            try:
                from src.ui.server import broadcast_state_threadsafe
                broadcast_state_threadsafe()
            except Exception:
                pass

            # On Linux / Raspberry Pi, prefer `aplay` to avoid PortAudio ALSA contention
            # and prevent HDMI clock re-sync which blanks the 7-inch display.
            if sys.platform.startswith("linux"):
                import shutil
                if shutil.which("aplay"):
                    import tempfile
                    import subprocess
                    import wave
                    tmp_wav = None
                    try:
                        int16_arr = np.clip(audio_arr * 32767.0, -32768, 32767).astype(np.int16)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                            tmp_wav = tf.name

                        with wave.open(tmp_wav, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(config.TTS_SAMPLE_RATE)
                            wf.writeframes(int16_arr.tobytes())

                        devices_to_try = self._find_best_alsa_devices()
                        if not devices_to_try:
                            if getattr(config, "ALLOW_HDMI_AUDIO", False):
                                devices_to_try = ["default"]
                            else:
                                config.log_debug("[speech] no non-HDMI audio device found; skipping HDMI to prevent screen blanking")

                        for dev in devices_to_try:
                            cmd = ["aplay", "-q", "-D", dev, tmp_wav] if dev != "default" else ["aplay", "-q", tmp_wav]
                            proc = subprocess.run(cmd, capture_output=True, timeout=30)
                            if proc.returncode == 0:
                                played = True
                                config.log_debug(f"[speech] played audio via {dev}")
                                break
                            else:
                                config.log_debug(f"[speech] aplay on {dev} returned {proc.returncode}: {proc.stderr.decode('utf-8', errors='ignore')}")
                    except Exception as ap_err:
                        config.log_debug(f"[speech] aplay error: {ap_err}")
                    finally:
                        if tmp_wav and os.path.exists(tmp_wav):
                            try:
                                os.remove(tmp_wav)
                            except Exception:
                                pass

            # Non-Linux fallback (e.g. macOS development)
            if not played and not sys.platform.startswith("linux"):
                try:
                    sd.play(audio_arr, config.TTS_SAMPLE_RATE)
                    sd.wait()
                    played = True
                except Exception as e:
                    try:
                        import scipy.signal
                        resampled = scipy.signal.resample_poly(audio_arr, 441, 240)
                        sd.play(resampled, 44100)
                        sd.wait()
                        played = True
                    except Exception as e2:
                        print(f"[speech] playback note: {e} (resample retry: {e2})", file=sys.stderr)

            if not played:
                # Visual fallback: keep mouth animation running for the duration of the speech
                duration = max(1.5, len(audio_arr) / config.TTS_SAMPLE_RATE)
                t0 = time.time()
                while time.time() - t0 < duration:
                    if self.interrupt_event and self.interrupt_event.is_set():
                        break
                    time.sleep(0.05)
        except Exception as e:
            config.log_debug(f"[speech] playback error: {e}")
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
            if audio is not None and not (self.interrupt_event and self.interrupt_event.is_set()):
                played = self._play_audio(audio)
        except Exception as e:
            print(f"[speech] TTS speak error: {e}")
        finally:
            if self.speaking_event:
                self.speaking_event.clear()
        return played
