"""
Mic -> continuous streaming speech detection -> Groq / Whisper transcription -> working memory.
Ultra-sensitive, non-chopping speech-to-text with generous silence thresholds and pre-buffering.
"""
import io
import queue
import wave
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from groq import Groq

import config

SAMPLE_RATE = 16000
BLOCK_SIZE = 1600  # 100ms blocks per audio chunk

# Common Whisper hallucinations when transcribing near-silence or ambient noise
HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching.", "thanks for watching!",
    "subtitles by", "amara.org", "subscribe", "bye.", "you", "okay.", "so",
    "thank you for watching.", "listening", "unbelievable.", "foreign",
    "i'm sorry.", "i'm sorry", "sorry.", "sorry", "aachman! no?", "aachman",
    "see my last thing okay so", "aachman! no", "aachman no", "thank you for listening."
}


def audio_to_wav_bytes(audio_np, sample_rate=SAMPLE_RATE):
    """Converts normalized float32 numpy audio array to 16-bit PCM WAV byte string."""
    pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    buf.seek(0)
    return buf.read()


def is_valid_transcript(text):
    if not text or len(text) < config.MIN_TRANSCRIPT_CHARS:
        return False
    lower = text.lower().strip().strip(".").strip("!").strip("?")
    if lower in HALLUCINATIONS or text.lower().strip() in HALLUCINATIONS:
        return False
    for bad in ["subtitles by", "amara.org", "thanks for watching", "thank you for watching", "i'm sorry", "aachman"]:
        if bad in lower:
            return False
    return True


def transcribe_audio(audio_np, groq_client, local_whisper):
    """Transcribes audio using Groq Whisper Large V3 Turbo with fallback to local Whisper."""
    if groq_client and getattr(config, "STT_BACKEND", "groq") == "groq":
        try:
            wav_bytes = audio_to_wav_bytes(audio_np)
            resp = groq_client.audio.transcriptions.create(
                file=("speech.wav", wav_bytes),
                model=getattr(config, "GROQ_STT_MODEL", "whisper-large-v3-turbo"),
                language="en",
                temperature=0.0,
            )
            text = resp.text.strip()
            if is_valid_transcript(text):
                return text
        except Exception as e:
            print(f"[audio] Groq STT error: {e}, using local fallback...")

    # Fallback to local faster-whisper
    if local_whisper:
        try:
            segments, _ = local_whisper.transcribe(
                audio_np,
                language="en",
                vad_filter=True,
                beam_size=1,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if is_valid_transcript(text):
                return text
        except Exception as e:
            print(f"[audio] Local STT error: {e}")

    return None


# Load Silero VAD neural model for precise voice activity detection
silero_vad_model = None
try:
    import torch
    silero_vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    print("[audio] Silero VAD neural voice activity detector initialized!")
except Exception as e:
    print(f"[audio] Silero VAD fallback to energy gate: {e}")


def run_audio(memory, stop_event, speaking_event=None):
    groq_client = None
    if getattr(config, "GROQ_API_KEY", None):
        try:
            groq_client = Groq(api_key=config.GROQ_API_KEY)
            print("[audio] STT engine: Groq (whisper-large-v3-turbo)")
        except Exception as e:
            print(f"[audio] Groq init failed: {e}")

    print("[audio] loading local whisper fallback model...")
    local_whisper = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    run_audio_loop(memory, stop_event, groq_client, local_whisper, speaking_event)


def run_audio_loop(memory, stop_event, groq_client=None, local_whisper=None, speaking_event=None):
    q = queue.Queue()

    grace_frames_needed = int((getattr(config, "VAD_POST_SPEECH_GRACE_MS", 300) / 1000.0) * SAMPLE_RATE / BLOCK_SIZE)
    # Use a list/dict to hold mutable state across the closure
    state = {"grace_frames_remaining": 0, "was_speaking": False}

    def callback(indata, frames, time_info, status):
        # Hardware-level hard mute when agent is speaking
        if speaking_event and speaking_event.is_set():
            state["was_speaking"] = True
            state["grace_frames_remaining"] = grace_frames_needed
            return

        if state["was_speaking"]:
            state["was_speaking"] = False

        # Post-speech grace period to let acoustic echo decay
        if state["grace_frames_remaining"] > 0:
            state["grace_frames_remaining"] -= 1
            return

        q.put(indata.copy())
    print("[audio] ready -- streaming microphone audio continuously with Silero VAD")

    # Ultra-fast speech detection parameters
    bg_energy = 0.002
    alpha = 0.95
    SPEECH_MULT = 2.2         # Backup energy gate multiplier
    SILENCE_TIMEOUT = 0.5      # 0.5s silence threshold to trigger STT instantly
    MIN_SPEECH_DURATION = 0.2  # 0.2s minimum duration to capture fast words

    pre_buffer_max = int(0.8 * SAMPLE_RATE)  # 0.8 seconds pre-speech buffer padding
    pre_buffer = np.zeros(0, dtype=np.float32)

    speech_buffer = []
    is_speaking = False
    silence_duration = 0.0
    zero_frame_count = 0

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=BLOCK_SIZE, callback=callback):
            while not stop_event.is_set():
                if speaking_event and speaking_event.is_set():
                    # Drain queue while agent is speaking to prevent self-transcription
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
                    is_speaking = False
                    memory.set_user_speaking(False)
                    speech_buffer = []
                    pre_buffer = np.zeros(0, dtype=np.float32)
                    silence_duration = 0.0
                    if silero_vad_model is not None:
                        try:
                            silero_vad_model.reset_states()
                        except Exception:
                            pass
                    stop_event.wait(0.2)
                    continue

                try:
                    data = q.get(timeout=0.2).flatten()
                except queue.Empty:
                    continue

                if data.max() == 0.0:
                    zero_frame_count += 1
                    if zero_frame_count == 30:
                        print("[audio] WARNING: Microphone audio is 0.0 (silent). "
                              "Please enable Microphone permission for your terminal app in "
                              "macOS System Settings > Privacy & Security > Microphone.")
                else:
                    zero_frame_count = 0

                energy = np.abs(data).mean()

                # Audio salience (loud noise spike)
                if energy > bg_energy * 10 and energy > 0.05:
                    memory.add(kind="conscious_trigger", text="Loud noise detected!", salience=0.9)

                if not is_speaking:
                    bg_energy = alpha * bg_energy + (1 - alpha) * energy
                    pre_buffer = np.concatenate((pre_buffer, data))
                    if len(pre_buffer) > pre_buffer_max:
                        pre_buffer = pre_buffer[-pre_buffer_max:]

                threshold = max(bg_energy * SPEECH_MULT, 0.005)

                # Evaluate Voice Activity with Silero VAD or Energy fallback
                is_speech = False
                if silero_vad_model is not None:
                    try:
                        import torch
                        audio_tensor = torch.from_numpy(data)
                        prob = silero_vad_model(audio_tensor, SAMPLE_RATE).item()
                        is_speech = (prob > 0.40)
                    except Exception:
                        is_speech = (energy > threshold)
                else:
                    is_speech = (energy > threshold)

                if is_speech:
                    if not is_speaking:
                        is_speaking = True
                        memory.set_user_speaking(True)
                        # Prepend full 1.2s pre-speech buffer so sentence prefixes are never cut!
                        speech_buffer = [pre_buffer]
                        pre_buffer = np.zeros(0, dtype=np.float32)
                    speech_buffer.append(data)
                    silence_duration = 0.0
                elif is_speaking:
                    speech_buffer.append(data)
                    silence_duration += (len(data) / SAMPLE_RATE)
                    if silence_duration >= SILENCE_TIMEOUT:
                        full_audio = np.concatenate(speech_buffer).astype(np.float32)
                        speech_buffer = []
                        is_speaking = False
                        memory.set_user_speaking(False)
                        silence_duration = 0.0

                        duration = len(full_audio) / SAMPLE_RATE
                        if duration < MIN_SPEECH_DURATION:
                            continue

                        # Amplify quiet audio for optimal transcription
                        max_val = np.abs(full_audio).max()
                        if max_val > 0 and max_val < 0.1:
                            full_audio = full_audio * (0.3 / max_val)
                        full_audio = np.clip(full_audio, -1.0, 1.0)

                        print(f"[audio] processing {duration:.1f}s speech...")
                        text = transcribe_audio(full_audio, groq_client, local_whisper)
                        if text:
                            print(f"[audio] heard: '{text}'")
                            memory.add(kind="speech", text=text)
    except Exception as e:
        print(f"[audio] mic stream error: {e}")
