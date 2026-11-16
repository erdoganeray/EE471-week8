"""MediaPipe Audio Transcriber: on-device speech-to-text (Parakeet TDT 0.6B) for a wav file."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
import numpy as np
import soundfile as sf
from common import ASSETS, MODELS, base_options

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--input", default=str(ASSETS / "audio" / "speech_16000_hz_mono.wav"))
args = ap.parse_args()

samples, sr = sf.read(args.input, dtype="float32")
if samples.ndim > 1:
    samples = samples.mean(axis=1)
if sr != 16000:  # model expects 16 kHz; simple linear resample
    n = int(len(samples) * 16000 / sr)
    samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples).astype("float32")
    sr = 16000
print(f"{args.input}: {len(samples) / sr:.1f} s")

cache = MODELS / "cache"
cache.mkdir(exist_ok=True)
opts = mp.tasks.audio.AudioTranscriberOptions(
    base_options=base_options("parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm"),
    sample_rate_hz=sr, cache_dir=str(cache))
with mp.tasks.audio.AudioTranscriber.create_from_options(opts) as asr:
    result = asr.transcribe(samples, end_of_stream=True)

for seg in result.segments:
    print(f"[{seg.from_timestamp_ms / 1000:5.1f}s - {seg.to_timestamp_ms / 1000:5.1f}s] {seg.transcript}")
