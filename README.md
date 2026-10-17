# Embedded Gesture AI

Live webcam prototype that draws hand landmarks, labels a few static hand poses,
and marks the index fingertip with its pixel coordinates in the video.

## Run

1. Install the project dependencies:

   ```bash
   uv sync
   ```

2. Download the MediaPipe hand landmark model into `models/`:

   ```bash
   mkdir -p models
   curl -L \
     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
     -o models/hand_landmarker.task
   ```

3. Start the webcam demo:

   ```bash
   uv run embedded-gesture-ai
   ```

Press **Q** in the video window to quit. Use `--camera 1` to try a different
camera, or `--model path/to/hand_landmarker.task` to select another model file.

The initial pose labels are **Point**, **Peace**, **Open palm**, **Fist**, and
**Thumbs up**. These are simple geometric classifications for upright, static
poses, not full sign-language recognition or translation. The pointer shows
where the index fingertip appears in the camera image; it does not estimate
the direction of a 3D point or control the computer cursor.