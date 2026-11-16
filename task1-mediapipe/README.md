# Task 1 - MediaPipe solutions in Python

One small script per MediaPipe solution (22 in total). All run on-device with `mediapipe 1.2.3`, no API keys.

## Setup
```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python download_models.py --list  # shows what is missing
python download_models.py         # small models (<100 MB)
python download_models.py --large # also the big .litertlm / .task models (~2 GB)
```
`download_models.py` is resumable (`.part` files + HTTP Range), skips finished files and downloads smallest first, so it is safe on a slow or flaky connection. Models go to `models/` (git-ignored).

## Scripts
Run from this folder, e.g. `python vision/face_detector.py` (most accept `--input <file>`; text ones accept the text as arguments).

| # | Solution | Script | Model | Demo input / output |
|---|----------|--------|-------|---------------------|
| 1 | Audio Classifier | `audio/audio_classifier.py` | YAMNet | speech wav -> `Speech` per window |
| 2 | Audio Transcriber | `audio/audio_transcriber.py` | Parakeet TDT 0.6B (615 MB) | speech wav -> transcript |
| 3 | Text-to-Speech | `audio/text_to_speech.py` | Phonon 31M (87 MB) | text -> `outputs/tts_output.wav` |
| 4 | Decision Maker | `text/decision_maker.py` | Laya s256 (678 MB) | boolean / choice / score answers |
| 5 | Semantic Retriever | `text/semantic_retriever.py` | EmbeddingGemma 2 text 270M | query -> best documents |
| 6 | Universal Embedder | `text/universal_embedder.py` | EmbeddingGemma 2 text+vision 440M | text vs image similarity matrix |
| 7 | Language Detector | `text/language_detector.py` | language_detector | sentence -> `en/tr/de/fr/es` |
| 8 | Text Classifier | `text/text_classifier.py` | BERT | sentence -> positive/negative |
| 9 | Text Embedder | `text/text_embedder.py` | Universal Sentence Encoder | cosine similarity |
| 10 | Text Proofreader | `text/text_proofreader.py` | proofread 200M (LiteRT-LM) | typos -> corrected text |
| 11 | Text Summarizer | `text/text_summarizer.py` | summarization 200M | KEYPOINTS and TLDR |
| 12 | Face Detector | `vision/face_detector.py` | BlazeFace short range | `outputs/face_detector.png` |
| 13 | Face Landmarker | `vision/face_landmarker.py` | face_landmarker | 478 points, blendshapes, head pose |
| 14 | Gesture Recognizer | `vision/gesture_recognizer.py` | gesture_recognizer | `Thumb_Up` |
| 15 | Hand Landmarker | `vision/hand_landmarker.py` | hand_landmarker | 21 points per hand |
| 16 | Holistic Landmarker | `vision/holistic_landmarker.py` | holistic_landmarker | face + pose + hands |
| 17 | Image Classifier | `vision/image_classifier.py` | EfficientNet-Lite0 | burger -> `cheeseburger` |
| 18 | Image Embedder | `vision/image_embedder.py` | MobileNet-V3 small | image similarity |
| 19 | Image Segmenter | `vision/image_segmenter.py` | selfie_segmenter | blurred background |
| 20 | Interactive Segmenter | `vision/interactive_segmenter.py` | MagicTouch | mask from a simulated click |
| 21 | Object Detector | `vision/object_detector.py` | EfficientDet-Lite0 | cat + dog boxes |
| 22 | Pose Landmarker | `vision/pose_landmarker.py` | pose_landmarker lite | 33 body points |

Annotated results are saved in `outputs/`.

## Notes
- `common.py` has the shared helpers (model lookup with a clear "model missing" message, image loading, drawing).
- Image Segmenter: the selfie model has a single label (`selfie`) and the person is category 0.
- Object Detector uses the official `cat_and_dog.jpg`; EfficientDet-Lite0/Lite2 mislabel the dogs in `cats_and_dogs.jpg` as cats.
- The new LiteRT-LM tasks write a cache to `models/cache/` on first use (git-ignored).
- Text-to-Speech output can be fed to the Audio Transcriber (round trip works: "Hello. This is a test of MediaPipe text to speech running on my laptop.").
