"""
Text-to-Speech Subsystem ("The Voice").
High-quality, in-process local speech synthesis via 4-bit Quantized Kokoro-82M ONNX.
"""
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


def clean_for_speech(text: str) -> str:
    """Pre-process text for natural speech synthesis, stripping markdown and bracketed tags."""
    if not text:
        return ""

    # Strip asterisks (e.g. *chuckles*, *smiles*)
    text = re.sub(r"\*.*?\*", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w\s.,!?'\-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class TTSEngine:
    """Coordinates Kokoro Q4 quantized ONNX synthesis and sounddevice hardware playback."""

    def __init__(self, lang_code: str = config.TTS_LANG_CODE, voice: str = config.TTS_VOICE,
                 speaking_event: Optional[threading.Event] = None,
                 interrupt_event: Optional[threading.Event] = None):
        self.voice = voice
        self.speaking_event = speaking_event
        self.interrupt_event = interrupt_event
        self._synth_lock = threading.Lock()
        self.onnx_session = None
        self._voices_cache: Dict[str, np.ndarray] = {}

        model_path = getattr(config, "KOKORO_MODEL_PATH", "")
        voices_path = getattr(config, "KOKORO_VOICES_PATH", "")

        # 1. Initialize tokenizer
        from kokoro import KPipeline
        self.pipeline = KPipeline(lang_code=lang_code)

        # 2. Try loading 4-bit Quantized ONNX model
        if model_path and os.path.exists(model_path) and voices_path and os.path.exists(voices_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = getattr(config, "N_THREADS", 4)
                opts.inter_op_num_threads = 1
                self.onnx_session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
                self._load_voice_tensor(voices_path, self.voice)
                print(f"[speech] initialized 4-bit Quantized Kokoro-82M ONNX model from {model_path} (Ultra-low RAM)!")
            except Exception as e:
                print(f"[speech] ONNX Q4 init note: {e}")
                self.onnx_session = None

        # Warmup pipeline on startup to eliminate cold-start delay
        try:
            self._synthesize("warmup", speed=1.0)
        except Exception:
            pass

    def _load_voice_tensor(self, voices_path: str, voice_name: str) -> Optional[np.ndarray]:
        """Loads a single voice embedding tensor from voices-v1.0.bin."""
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
            print(f"[speech] voice load error: {e}")
        return None

    def _synthesize(self, text: str, speed: float = 1.0) -> Optional[np.ndarray]:
        """Synthesize text chunk to float32 numpy array."""
        spoken = clean_for_speech(text)
        if not spoken:
            return None

        with self._synth_lock:
            try:
                # High-speed Quantized ONNX path (Q4 quantized)
                if self.onnx_session is not None:
                    voices_path = getattr(config, "KOKORO_VOICES_PATH", "")
                    voice_arr = self._load_voice_tensor(voices_path, self.voice)
                    if voice_arr is not None:
                        ps, _ = self.pipeline.g2p(spoken)
                        if ps:
                            input_ids = np.array([[0] + [self.pipeline.vocab[c] for c in ps if c in self.pipeline.vocab] + [0]], dtype=np.int64)
                            tokens_len = len(input_ids[0])
                            style = np.ascontiguousarray(voice_arr[min(tokens_len, len(voice_arr) - 1)], dtype=np.float32)
                            speed_arr = np.array([float(speed)], dtype=np.float32)
                            waveform = self.onnx_session.run(None, {
                                "input_ids": input_ids,
                                "style": style,
                                "speed": speed_arr
                            })[0]
                            if waveform is not None:
                                return waveform.flatten().astype(np.float32)

                # PyTorch fallback path
                chunks: List[np.ndarray] = []
                for _, _, audio in self.pipeline(spoken, voice=self.voice, speed=speed):
                    if audio is not None:
                        if hasattr(audio, "detach"):
                            arr = audio.detach().cpu().numpy().flatten().astype(np.float32)
                        else:
                            arr = np.asarray(audio, dtype=np.float32).flatten()
                        if len(arr) > 0:
                            chunks.append(arr)
                if chunks:
                    return np.concatenate(chunks).astype(np.float32)
            except Exception as e:
                print(f"[speech] synthesis error: {e}")
        return None

    def _play_audio(self, audio: Optional[np.ndarray]) -> None:
        """Play a pre-synthesized audio chunk through sounddevice."""
        if audio is None or len(audio) == 0:
            return
        if self.interrupt_event and self.interrupt_event.is_set():
            return

        try:
            audio_arr = np.ascontiguousarray(audio, dtype=np.float32)
            internal_state.set_playing_audio(True)
            sd.stop()
            sd.play(audio_arr, config.TTS_SAMPLE_RATE)
            sd.wait()
        except Exception as e:
            print(f"[speech] playback error: {e}")
        finally:
            internal_state.set_playing_audio(False)

    def speak(self, text: str, speed: float = 1.0) -> None:
        """Synthesize full text and play it in a single blocking call."""
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
