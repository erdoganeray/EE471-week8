"""Resumable, cache-friendly model downloader for the Task 1 MediaPipe demos.

Designed for a slow/flaky connection:
  * downloads to <file>.part and resumes with HTTP Range after any failure/Ctrl+C
  * skips files that are already complete (safe to re-run any time)
  * retries with backoff forever-ish (--retries), never restarts from zero
  * small models by default; big ones only when asked (--large / --only NAME)
  * can run in the background; progress is printed as plain lines (log friendly)

Usage:
  python download_models.py --list
  python download_models.py                  # all small models (< --max-mb, default 100)
  python download_models.py --group vision
  python download_models.py --only audio_transcriber
  python download_models.py --large          # also the big .litertlm models
"""
import argparse
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
GCS = "https://storage.googleapis.com/mediapipe-models"
HF = "https://huggingface.co/litert-community"

# name: (group, filename, url, approx size MB)
MODELS = {
    # vision
    "face_detector": ("vision", "blaze_face_short_range.tflite", f"{GCS}/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite", 1),
    "face_landmarker": ("vision", "face_landmarker.task", f"{GCS}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", 4),
    "gesture_recognizer": ("vision", "gesture_recognizer.task", f"{GCS}/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task", 8),
    "hand_landmarker": ("vision", "hand_landmarker.task", f"{GCS}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", 8),
    "holistic_landmarker": ("vision", "holistic_landmarker.task", f"{GCS}/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task", 20),
    "image_classifier": ("vision", "efficientnet_lite0.tflite", f"{GCS}/image_classifier/efficientnet_lite0/int8/latest/efficientnet_lite0.tflite", 5),
    "image_embedder": ("vision", "mobilenet_v3_small.tflite", f"{GCS}/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite", 4),
    "image_segmenter": ("vision", "selfie_segmenter.tflite", f"{GCS}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite", 1),
    "interactive_segmenter": ("vision", "interactive_segmentation.task", f"{GCS}/interactive_segmenter_v2/magic_touch/int8/latest/interactive_segmentation.task", 10),
    "object_detector": ("vision", "efficientdet_lite0.tflite", f"{GCS}/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite", 4),
    "pose_landmarker": ("vision", "pose_landmarker_lite.task", f"{GCS}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task", 6),
    # text
    "language_detector": ("text", "language_detector.tflite", f"{GCS}/language_detector/language_detector/float32/1/language_detector.tflite", 1),
    "text_classifier": ("text", "bert_classifier.tflite", f"{GCS}/text_classifier/bert_classifier/float32/latest/bert_classifier.tflite", 25),
    "text_embedder": ("text", "universal_sentence_encoder.tflite", f"{GCS}/text_embedder/universal_sentence_encoder/float32/latest/universal_sentence_encoder.tflite", 6),
    "text_proofreader": ("text", "proofread_quant_200m.litertlm", f"{GCS}/text_proofreader/200m/latest/proofread_quant_200m.litertlm", 200),
    "text_summarizer": ("text", "summarization_quant_200m_2modes.litertlm", f"{GCS}/text_summarizer/200m/latest/summarization_quant_200m_2modes.litertlm", 200),
    "decision_maker": ("text", "laya_s256.task", f"{GCS}/decision_maker/laya/float32/laya_s256/latest/laya_s256.task", 100),
    "semantic_retriever": ("text", "embeddinggemma-2-text-270m.litertlm", f"{HF}/embeddinggemma-2-text-270m-litert-lm/resolve/main/embeddinggemma-2-text-270m.litertlm", 300),
    "universal_embedder": ("text", "embeddinggemma-2-text-vision-440m.litertlm", f"{HF}/embeddinggemma-2-text-vision-440m-litert-lm/resolve/main/embeddinggemma-2-text-vision-440m.litertlm", 450),
    # audio
    "audio_classifier": ("audio", "yamnet.tflite", f"{GCS}/audio_classifier/yamnet/float32/latest/yamnet.tflite", 4),
    "audio_transcriber": ("audio", "parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm", f"{HF}/parakeet-tdt-0.6b-v3/resolve/main/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm", 615),
    "text_to_speech": ("audio", "phonon_int8.litertlm", f"{HF}/phonon-31m-en/resolve/main/phonon_int8.litertlm", 87),
}
# sizes are rough estimates from the docs; real size comes from Content-Length.


def fetch(url: str, dest: Path, retries: int) -> bool:
    # download into <file>.part; if the connection drops, retry and continue from the bytes we already have
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "ee471-downloader"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                status = r.status
                if have and status == 200:  # server ignored Range: restart
                    have = 0
                    part.unlink(missing_ok=True)
                total = have + int(r.headers.get("Content-Length", 0) or 0)
                mode = "ab" if have else "wb"
                last = time.time()
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 5:
                            last = time.time()
                            pct = f" ({have * 100 // total}%)" if total else ""
                            print(f"    {dest.name}: {have / 1e6:.1f} MB{pct}", flush=True)
                if total and have < total:
                    raise IOError(f"incomplete: {have}/{total} bytes")
            part.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 416:  # range past end: file is actually complete
                part.replace(dest)
                return True
            print(f"    HTTP {e.code} on attempt {attempt}/{retries}", flush=True)
            if e.code in (401, 403, 404):
                return False
        except Exception as e:  # network drop, timeout, ...
            print(f"    {type(e).__name__}: {e} (attempt {attempt}/{retries}); resuming", flush=True)
        time.sleep(min(5 * attempt, 30))
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--group", choices=["vision", "text", "audio"])
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--large", action="store_true", help="include models above --max-mb")
    ap.add_argument("--max-mb", type=int, default=100)
    ap.add_argument("--retries", type=int, default=30)
    a = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    # decide which models to download: --only NAME, or everything under --max-mb (or all with --large)
    if a.only:
        unknown = [n for n in a.only if n not in MODELS]
        if unknown:
            print("unknown:", unknown, "known:", list(MODELS))
            return 2
        names = a.only
    else:
        names = [n for n, (g, _, _, mb) in MODELS.items()
                 if (not a.group or g == a.group) and (a.large or mb <= a.max_mb)]
    names.sort(key=lambda n: MODELS[n][3])  # smallest first

    if a.list:
        for n, (g, f, _, mb) in MODELS.items():
            st = "ok " if (MODELS_DIR / f).exists() else "-- "
            print(f"{st}{n:24s} {g:7s} ~{mb:4d} MB  {f}")
        return 0

    # download one by one; finished files are skipped, so re-running is always safe
    failed = []
    for n in names:
        g, fname, url, mb = MODELS[n]
        dest = MODELS_DIR / fname
        if dest.exists():
            print(f"[skip] {n} already downloaded", flush=True)
            continue
        print(f"[get ] {n} (~{mb} MB)", flush=True)
        if fetch(url, dest, a.retries):
            print(f"[done] {n}", flush=True)
        else:
            print(f"[FAIL] {n}", flush=True)
            failed.append(n)
    print("failed:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
