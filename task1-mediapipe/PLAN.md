# Task 1 Plan: MediaPipe demos (22 solutions)

## Layout
```
task1-mediapipe/
  requirements.txt        mediapipe, numpy, opencv-python, pillow, soundfile
  download_models.py      fetches every model into models/ (git-ignored)
  common.py               model path helper, image/audio loaders, result drawing
  audio/   audio_classifier.py  audio_transcriber.py  text_to_speech.py
  text/    language_detector.py text_classifier.py text_embedder.py text_proofreader.py
           text_summarizer.py   decision_maker.py   semantic_retriever.py universal_embedder.py
  vision/  face_detector.py face_landmarker.py gesture_recognizer.py hand_landmarker.py
           holistic_landmarker.py image_classifier.py image_embedder.py image_segmenter.py
           interactive_segmenter.py object_detector.py pose_landmarker.py
  assets/  audio/ images/ text/   small sample inputs
  outputs/ annotated images / wavs (small ones committed)
```
One script per task, each runnable as `python <group>/<name>.py [--input ...]`, printing results and saving an annotated output. Naming matches recordings (`audio_classifier.avi`, ...).

## Groups and order
1. **Vision (11)** - easiest, reuse Task 2/3 images plus a few extras (objects, hands, gesture).
   Face Detector, Face Landmarker, Pose Landmarker, Hand Landmarker, Gesture Recognizer, Holistic Landmarker, Image Classifier, Image Embedder (cosine similarity of two images), Image Segmenter, Interactive Segmenter (point/scribble strokes), Object Detector.
2. **Text, classic (4)** - Language Detector, Text Classifier (BERT sentiment), Text Embedder (similarity), Universal Embedder.
3. **Text, new/LLM-style (4)** - Text Proofreader, Text Summarizer (.litertlm, Gemma terms), Decision Maker, Semantic Retriever (EmbeddingGemma 2).
4. **Audio (3)** - Audio Classifier (YAMNet, a wav clip), Audio Transcriber (~615 MB model), Text-to-Speech (Kokoro ~295 MB or smaller Phonon ~87 MB).

## Facts from the docs (checked today)
- All 22 have a Python section; `pip install mediapipe`, API as `mp.tasks.<domain>.<Class>`. Audio Transcriber and Text-to-Speech also need numpy.
- Newer tasks (Audio Transcriber, TTS, Decision Maker, Semantic Retriever, Universal Embedder, Text Proofreader, Text Summarizer) use `.litertlm`/`.task` models; the exact class names above come from the pages and may differ in the installed wheel.
- No API keys needed; everything runs on-device. Model URLs are on the solution pages and will go in `download_models.py`.

## Risks / checks before coding
- The newest tasks may only exist in a recent mediapipe release. First step: `pip install -U mediapipe` in a venv and check each class actually imports. Any missing one gets reported to you, not faked.
- Big model files (several hundred MB for transcriber/TTS/embedding) are downloaded, never committed.
- Windows: audio playback/saving uses soundfile (wav), no extra system deps.
- Fallback if a new task fails on Windows (e.g. needs Linux/GPU): run it in Docker/WSL and note it in the README.

## Deliverables per task
Script + sample input + saved output + short README section (what it does, key code blocks to explain in your recording). You record `<task_name>.avi` for each of the 22.

## Proposed steps
1. venv + install mediapipe, verify all 22 classes import.
2. `download_models.py` and `common.py`.
3. Implement and run in the group order above, committing after each group.
4. Fill README with a table (task, script, model, output, recording link) and a recording checklist for you.
