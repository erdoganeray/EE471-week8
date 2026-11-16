"""MediaPipe Text-to-Speech: synthesize speech on-device and save it as a wav file (Phonon 31M)."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
import soundfile as sf
from common import MODELS, base_options, out_path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--text", default="Hello! This is a test of MediaPipe text to speech running on my laptop.")
ap.add_argument("--output", default="tts_output.wav")
args = ap.parse_args()

cache = MODELS / "cache"
cache.mkdir(exist_ok=True)
opts = mp.tasks.audio.TextToSpeechOptions(
    base_options=base_options("phonon_int8.litertlm"), cache_dir=str(cache))
with mp.tasks.audio.TextToSpeech.create_from_options(opts) as tts:
    voices = tts.list_voices()
    print("voices:", [f"{v.voice_id} ({v.locale})" for v in voices])
    result = tts.synthesize(args.text)

path = out_path(args.output)
sf.write(path, result.samples, result.sample_rate_hz)
print(f"{len(result.samples) / result.sample_rate_hz:.1f} s of audio @ {result.sample_rate_hz} Hz -> {path}")
