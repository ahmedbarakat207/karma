#!/usr/bin/env python3
"""Diagnostic script to inspect and test audio devices and TTS playback on Raspberry Pi.

Run on your Raspberry Pi:
    python3 scripts/test_audio.py
"""

import os
import sys
import shutil
import subprocess
import time
import numpy as np

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print("=" * 65)
print("  KARMA AUDIO & TTS HARDWARE DIAGNOSTIC")
print("=" * 65)
print(f"Platform: {sys.platform} ({os.uname().machine if hasattr(os, 'uname') else 'unknown'})")
print(f"Python:   {sys.version.split()[0]}")

# 1. Inspect ALSA Devices
print("\n[1] ALSA Playback Hardware Devices (aplay -l):")
if shutil.which("aplay"):
    try:
        p = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=3)
        if p.stdout.strip():
            for line in p.stdout.strip().splitlines():
                if line.startswith("card") or "Subdevices" in line:
                    print(f"    {line}")
        else:
            print("    (No ALSA playback devices found)")
    except Exception as e:
        print(f"    Error running aplay: {e}")
else:
    print("    'aplay' not found in PATH")

# 2. Inspect Sound Servers (PulseAudio / PipeWire)
print("\n[2] Sound Servers:")
for server, cmd in [("PulseAudio", ["pactl", "info"]), ("PipeWire", ["wpctl", "status"])]:
    if shutil.which(cmd[0]):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if p.returncode == 0:
                print(f"    ✅ {server} is ACTIVE")
            else:
                print(f"    ⚪ {server} installed but not active ({p.stderr.strip()[:40]})")
        except Exception as e:
            print(f"    ⚪ {server} check error: {e}")
    else:
        print(f"    ⚪ {server} not installed")

# 3. PortAudio / SoundDevice Query
print("\n[3] PortAudio (sounddevice) Devices:")
try:
    import sounddevice as sd
    devs = sd.query_devices()
    for idx, d in enumerate(devs):
        out_ch = d.get("max_output_channels", 0)
        if out_ch > 0:
            print(f"    [{idx}] {d['name']} (outputs: {out_ch}, default_samplerate: {d.get('default_samplerate')})")
except Exception as e:
    print(f"    sounddevice error: {e}")

# 4. Set Volume to 100%
print("\n[4] Unmuting & Setting Volume to 100% across all cards...")
from src.speech.tts import set_system_volume_max, TTSEngine
set_system_volume_max()
print("    ✅ Volume set to 100% and unmuted.")

# 5. Play a clean test chime
print("\n[5] Generating and playing a 1-second stereo test chime (44.1 kHz)...")
sr = 44100
duration = 0.8
t = np.linspace(0, duration, int(sr * duration), False)
# Major triad chord (C5 - E5 - G5 - C6)
chord = 0.25 * np.sin(2 * np.pi * 523.25 * t) + \
        0.25 * np.sin(2 * np.pi * 659.25 * t) + \
        0.25 * np.sin(2 * np.pi * 783.99 * t) + \
        0.25 * np.sin(2 * np.pi * 1046.50 * t)
env = np.exp(-3.0 * t)
tone = (chord * env).astype(np.float32)

from src.speech.tts import TTSEngine
tts = TTSEngine()
print("    Testing _play_audio with test tone...")
success = tts._play_audio(tone)
if success:
    print("    ✅ Test tone PLAYED SUCCESSFULLY!")
else:
    print("    ❌ Test tone failed to play on all backends.")

# 6. Test Real TTS Synthesis and Playback (with verbose error surfacing)
print("\n[6] Testing Bilingual TTS Engine (Speaking test phrase)...")
phrase = "Karma audio test: sound is working at 100 percent!"
print(f"    Synthesizing: '{phrase}'")
import contextlib
import io as _io

print(f"    en_pipeline present: {getattr(tts, 'en_pipeline', None) is not None}")
print(f"    interrupt_event set: {bool(getattr(tts, 'interrupt_event', None) and tts.interrupt_event.is_set())}")
if hasattr(tts, "_find_best_alsa_devices"):
    try:
        print(f"    ALSA candidates: {tts._find_best_alsa_devices()}")
    except Exception as _e:
        print(f"    ALSA candidates query failed: {_e}")
else:
    print("    ALSA candidates: (not available in this TTSEngine version — run 'aplay -l' output from [1] above)")
if hasattr(tts, "_find_best_pulse_sink"):
    try:
        print(f"    Pulse sink: {tts._find_best_pulse_sink()}")
    except Exception as _e:
        print(f"    Pulse sink query failed: {_e}")
else:
    print("    Pulse sink: (not available in this TTSEngine version — see 'pactl list sinks short')")

t0 = time.time()
_stderr_buf = _io.StringIO()
with contextlib.redirect_stderr(_stderr_buf):
    tts_success = tts.speak(phrase)
elapsed = time.time() - t0
_err_text = _stderr_buf.getvalue().strip()
if _err_text:
    for _line in _err_text.splitlines():
        print(f"    TTS ERROR: {_line}")

if tts_success:
    print(f"    ✅ TTS speech synthesis & playback SUCCEEDED in {elapsed:.2f}s!")
else:
    print("    ❌ TTS speech playback failed.")
    if not _err_text:
        print("    (no stderr from TTS engine — silent early-return; synthesis likely None/empty or interrupted)")

print("\n" + "=" * 65)
if success or tts_success:
    print("  AUDIO SUBSYSTEM IS WORKING!")
else:
    print("  TROUBLESHOOTING TIP:")
    print("  - Make sure headphones or speaker are plugged firmly into the 3.5mm jack or USB.")
    print("  - If using USB DAC, run: lsusb")
    print("  - Try setting an explicit device in .env, e.g.: AUDIO_OUTPUT_DEVICE=plughw:CARD=Headphones,DEV=0")
print("=" * 65)
