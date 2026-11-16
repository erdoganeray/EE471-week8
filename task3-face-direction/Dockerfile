FROM python:3.12-slim

# libs needed by the mediapipe wheels on Debian slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the face model is committed next to the script, so no download happens at build/run time
COPY predict.py face_landmarker.task ./

# usage: docker run --rm -v <host-dir>:/input face-direction /input/<image>
ENTRYPOINT ["python", "-W", "ignore", "predict.py"]
