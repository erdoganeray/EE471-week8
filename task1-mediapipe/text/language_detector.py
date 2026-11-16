"""MediaPipe Language Detector: language code + probability for each sentence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import base_options

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Bugun hava cok guzel, parka gidelim mi?",
    "Der schnelle braune Fuchs springt ueber den faulen Hund.",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "El rapido zorro marron salta sobre el perro perezoso.",
]
if len(sys.argv) > 1:
    SENTENCES = sys.argv[1:]

opts = mp.tasks.text.LanguageDetectorOptions(
    base_options=base_options("language_detector.tflite"), max_results=2)
with mp.tasks.text.LanguageDetector.create_from_options(opts) as det:
    for s in SENTENCES:
        res = det.detect(s)
        top = ", ".join(f"{d.language_code}={d.probability:.2f}" for d in res.detections)
        print(f"{top:22s} | {s}")
