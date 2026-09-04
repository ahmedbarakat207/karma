#!/usr/bin/env python3
import io
import os
import re
import threading
import time
import zipfile
from typing import Optional, List, Dict
import numpy as np
import sounddevice as sd

from src import config
from src.state import internal_state
from src.speech.arabic_g2p import is_arabic, ArabicG2P, EXTRA_SYMBOLS, clean_phonemes


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
        from kokoro import KPipeline
        self.en_pipeline = KPipeline(lang_code=lang_code)
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
            use_arabic = is_arabic(spoken) and getattr(config, "NABRA_ENABLED", True)
            if use_arabic:
                audio = self._synthesize_arabic(spoken, speed=speed)
                if audio is not None:
                    return audio
                # Fallback to English pipeline if Arabic synthesis failed
                return self._synthesize_english(spoken, speed=speed)
            else:
                return self._synthesize_english(spoken, speed=speed)

    def _play_audio(self, audio: Optional[np.ndarray]) -> None:
        if audio is None or len(audio) == 0:
            return
        if self.interrupt_event and self.interrupt_event.is_set():
            return

        try:
            audio_arr = np.ascontiguousarray(audio, dtype=np.float32)
            internal_state.set_playing_audio(True)
            sd.play(audio_arr, config.TTS_SAMPLE_RATE)
            sd.wait()
        except Exception as e:
            config.log_debug(f"[speech] playback error: {e}")
        finally:
            internal_state.set_playing_audio(False)

    def speak(self, text: str, speed: float = 1.0) -> None:
        spoken_text = clean_for_speech(text)
        if not spoken_text:
            return

        if self.interrupt_event:
            self.interrupt_event.clear()
        if self.speaking_event:
            self.speaking_event.set()

        try:
            audio = self._synthesize(spoken_text, speed=speed)
            if audio is not None and not (self.interrupt_event and self.interrupt_event.is_set()):
                self._play_audio(audio)
        except Exception as e:
            print(f"[speech] TTS speak error: {e}")
        finally:
            if self.speaking_event:
                self.speaking_event.clear()
