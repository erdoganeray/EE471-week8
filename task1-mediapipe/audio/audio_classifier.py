"""MediaPipe Audio Classifier: YAMNet sound-event labels for a wav clip."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
import soundfile as sf
from common import ASSETS, base_options

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--input", default=str(ASSETS / "audio" / "speech_16000_hz_mono.wav"))  # --input <wav>
args = ap.parse_args()

# Step 1: read the wav file: audio is just a list of numbers (samples) + the sample rate
samples, sample_rate = sf.read(args.input, dtype="float32", always_2d=False)
if samples.ndim > 1:  # stereo -> mono
    samples = samples.mean(axis=1)
print(f"{args.input}: {len(samples) / sample_rate:.1f} s @ {sample_rate} Hz")

# Step 2: choose the model (YAMNet, 521 sound classes) and keep the 3 best labels
opts = mp.tasks.audio.AudioClassifierOptions(
    base_options=base_options("yamnet.tflite"), max_results=3)

# Step 3: wrap the samples in MediaPipe's AudioData and classify
audio = mp.tasks.components.containers.AudioData.create_from_array(samples, sample_rate)
with mp.tasks.audio.AudioClassifier.create_from_options(opts) as clf:
    results = clf.classify(audio)  # one result per ~0.975 s window

# Step 4: print the top labels for every window
for i, r in enumerate(results):
    top = ", ".join(f"{c.category_name}={c.score:.2f}" for c in r.classifications[0].categories)
    print(f"window {i} (t={r.timestamp_ms} ms): {top}")
