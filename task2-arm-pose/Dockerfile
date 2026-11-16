FROM python:3.12-slim

# libs needed by the mediapipe/opencv wheels on Debian slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the pose model is committed next to the script, so no download happens at build/run time
COPY predict.py pose_landmarker_lite.task ./

# usage: docker run --rm -v <host-dir>:/input arm-pose /input/<image>
ENTRYPOINT ["python", "-W", "ignore", "predict.py"]
