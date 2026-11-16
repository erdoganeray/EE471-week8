# Task 2 - Which arm is raised?

`predict.py PATH` prints one of `left`, `right`, `both`, `None`.

## How it works
1. **Pose Landmarker** (MediaPipe, lite model) finds 33 body landmarks.
2. An arm counts as raised when its **wrist is above its shoulder** by more than 5 % of the torso length (image y grows downward); landmarks with visibility < 0.5 are ignored.
3. Left/right/both/None is decided from the two arms.

`left`/`right` = the **person's own** left/right arm (MediaPipe convention), so the arm that appears on the image's *right* side is `left`.

## Run locally
```bash
pip install -r requirements.txt
python predict.py pose-1.jpg
```

## Run with Docker
```bash
docker build -t arm-pose .
docker run --rm -v "$(pwd):/input" arm-pose /input/pose-1.jpg
```
(PowerShell: `-v "${PWD}:/input"`.) The model file is part of the image, nothing is downloaded at run time.

## Results
| image | output |
|-------|--------|
| pose-1.jpg | left |
| pose-2.jpg | right |
| pose-3.jpg | both |
