"""MediaPipe Text-to-Speech: synthesize speech on-device and save it as a wav file (Phonon 31M)."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
import soundfile as sf
from common import MODELS, base_options, out_path

# --text = what to say, --output = wav file name (saved in outputs/)
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--text", default="Hello! This is a test of MediaPipe text to speech running on my laptop.")
ap.add_argument("--output", default="tts_output.wav")
args = ap.parse_args()

# Step 1: choose the model (small 87 MB voice model)
cache = MODELS / "cache"  # files the model prepares on first run
cache.mkdir(exist_ok=True)
opts = mp.tasks.audio.TextToSpeechOptions(
    base_options=base_options("phonon_int8.litertlm"), cache_dir=str(cache))

with mp.tasks.audio.TextToSpeech.create_from_options(opts) as tts:
    # Step 2: list the voices inside the model (id + language)
    voices = tts.list_voices()
    print("voices:", [f"{v.voice_id} ({v.locale})" for v in voices])
    # Step 3: turn the text into audio samples (mono float32)
    result = tts.synthesize(args.text)

# Step 4: save the samples as a wav file you can play
path = out_path(args.output)
sf.write(path, result.samples, result.sample_rate_hz)
print(f"{len(result.samples) / result.sample_rate_hz:.1f} s of audio @ {result.sample_rate_hz} Hz -> {path}")
