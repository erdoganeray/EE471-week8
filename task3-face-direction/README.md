# Task 3 - Which way is the face looking?

`predict.py PATH` prints one of `left`, `right`, `straight`.

## How it works
1. **Face Landmarker** (MediaPipe) finds 478 face landmarks and a 4x4 *facial transformation matrix* (head pose).
2. The head **yaw** (rotation around the vertical axis) is computed from the rotation part of that matrix.
3. `|yaw| < 15°` -> `straight`; otherwise the sign decides: negative yaw -> `left`, positive -> `right`.

`left`/`right` is **as seen in the image** (face turned towards the image's left edge = `left`). To switch to the person's own left/right, swap the two returns in `classify()`.

If no face is found the script exits with an error message instead of guessing.

## Run locally
```bash
pip install -r requirements.txt
python predict.py face-1.png
```

## Run with Docker
```bash
docker build -t face-direction .
docker run --rm -v "$(pwd):/input" face-direction /input/face-1.png
```
(PowerShell: `-v "${PWD}:/input"`.) The model file is part of the image, nothing is downloaded at run time.

## Results
| image | yaw | output |
|-------|-----|--------|
| face-1.png | -34.7° | left |
| face-2.png | +32.5° | right |
| face-3.png | -2.6° | straight |
