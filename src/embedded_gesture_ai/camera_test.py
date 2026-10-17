''' This file takes the webcam frames and runs them through media pipe
Four jobs happening
1.) Get Video
2.) Detect hand landmarks
3.) Interperet the landmarks
4.) Draw result
'''

import argparse
import time #Gives every video a timestamp
from collections.abc import Sequence
from pathlib import Path

import cv2 # Handles webcam, drawing and displaying images
import mediapipe as mp # Handles AI hand detection

'''
Wrist: 0
Thumb: 1, 2, 3, 4
Index: 5, 6, 7, 8
Middle: 9, 10, 11, 12
Ring: 13, 14, 15, 16
Pinky: 17, 18, 19, 20
'''
LANDMARK_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

#Labels
FINGER_TIPS = (8, 12, 16, 20)
FINGER_PIPS = (6, 10, 14, 18)
INDEX_FINGER_TIP = 8
THUMB_TIP = 4
THUMB_IP = 3
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

''' MediaPipe -> detects landmarks, program decides if landmarks are present
Open palm
Point
Peace
Thumbs up
Fist
'''
def recognize_gesture(
    landmarks: Sequence[mp.tasks.components.containers.NormalizedLandmark],
    handedness: str,
) -> str:
    # Classify a few upright, static hand poses from their landmark positions
    if len(landmarks) != 21:
        return "Hand"

    extended = [
        landmarks[tip].y < landmarks[pip].y
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
    ]
    thumb_extended = (
        landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x
        if handedness == "Right"
        else landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x
        if handedness == "Left"
        else False
    )

    if all(extended):
        return "Open palm"
    if extended == [True, False, False, False]:
        return "Point"
    if extended == [True, True, False, False]:
        return "Peace"
    if not any(extended) and thumb_extended:
        return "Thumbs up"
    if not any(extended):
        return "Fist"
    return "Hand"

# Normailization for MediaPipe landmarks
def point_for_frame(
    landmark: mp.tasks.components.containers.NormalizedLandmark,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:

    # Turns co-ord range into pixel co-ords, while keeping it inside image bounds
    x = min(max(round(landmark.x * frame_width), 0), frame_width - 1) # Turns co-ord range into pixel co-ords
    y = min(max(round(landmark.y * frame_height), 0), frame_height - 1)

    return x, y

# Draws results of point_for_frame onto image
def draw_hand(
    frame: cv2.typing.MatLike,
    landmarks: Sequence[mp.tasks.components.containers.NormalizedLandmark],
    label: str,
) -> tuple[int, int]:
    height, width = frame.shape[:2]
    points = [point_for_frame(point, width, height) for point in landmarks]

    # For every connection OpenCV draws a line
    for start, end in LANDMARK_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (90, 220, 90), 2)
    # For every joint, OpenCV Draws a circle
    for point in points:
        cv2.circle(frame, point, 4, (255, 180, 40), -1)

    # Tracks tip of index finger
    pointer = points[INDEX_FINGER_TIP]
    # Draws cross red cross @ tip of index finger
    cv2.drawMarker(
        frame,
        pointer,
        (0, 80, 255),
        cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )
    cv2.putText(
        frame,
        f"{label} | tip ({pointer[0]}, {pointer[1]})",
        (max(10, points[0][0]), max(30, points[0][1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return pointer

# Main engine of program
def run_camera(camera_index: int, model_path: Path) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Hand landmark model not found at {model_path}. "
            "Download hand_landmarker.task using the instructions in README.md."
        )

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check camera permissions "
            "or choose another camera with --camera."
        )

    # MediaPipe Configuration
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    try:

        # Creating the landmarker
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            last_timestamp_ms = -1

            # Main camera loop
            while True:
                # Read one frame
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Camera stopped returning frames.")

                # Turns camera image into "mirror image"
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = max(time.monotonic_ns() // 1_000_000, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                ''' AI model runs here, MediaPipe does image processing here
                Input: camera frame
                Output: detected hand
                        landmarks
                        handedness
                '''
                result = landmarker.detect_for_video(image, timestamp_ms)

                if result.hand_landmarks:
                    for hand_index, landmarks in enumerate(result.hand_landmarks):
                        handedness = "Unknown"
                        if hand_index < len(result.handedness) and result.handedness[hand_index]:
                            handedness = result.handedness[hand_index][0].category_name
                        gesture = recognize_gesture(landmarks, handedness)

                        # Everything is drawn here
                        draw_hand(frame, landmarks, f"{handedness}: {gesture}")
                else:
                    cv2.putText(
                        frame,
                        "Show your hand to the camera",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frame,
                    "Press Q to quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Hand signs and pointing", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize a few hand poses and track the index fingertip."
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam device index")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="path to the MediaPipe hand_landmarker.task model",
    )
    args = parser.parse_args()
    run_camera(args.camera, args.model)


if __name__ == "__main__":
    main()
