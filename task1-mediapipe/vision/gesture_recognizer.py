"""MediaPipe Gesture Recognizer: hand gesture (Thumb_Up, Victory, ...) + landmarks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, draw_landmarks, save_rgb

args = default_args(__doc__, "thumbs_up.jpg")
opts = mp.tasks.vision.GestureRecognizerOptions(
    base_options=base_options("gesture_recognizer.task"), num_hands=2)
image = load_mp_image(args.input)
with mp.tasks.vision.GestureRecognizer.create_from_options(opts) as gr:
    result = gr.recognize(image)

rgb = to_rgb(image)
conns = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
print(f"hands: {len(result.hand_landmarks)}")
for hand, gest, handed in zip(result.hand_landmarks, result.gestures, result.handedness):
    print(f"  {handed[0].category_name} hand -> {gest[0].category_name} ({gest[0].score:.2f})")
    draw_landmarks(rgb, hand, conns)
save_rgb(rgb, "gesture_recognizer.png")
