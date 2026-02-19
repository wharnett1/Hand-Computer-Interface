"""
Hand gesture mouse control using the MediaPipe Tasks API (v0.10.32+).
Captures webcam video, detects hands, and translates gestures into mouse actions.

Variant: PIP TRACKING — uses the index finger PIP (middle knuckle) joint for
cursor movement instead of the fingertip. The PIP barely moves when pinching,
eliminating pre-pinch cursor drift at the source.

Gesture mapping:
  - Point (index only)          -> move cursor (relative, like a trackpad)
  - Thumb-index pinch           -> left click  (fires on pinch-down)
  - Pinch + move while held     -> click-and-drag
  - Thumb-middle-finger pinch   -> right click
  - Peace / Victory + move      -> scroll up/down
  - Fist                        -> double click (fires once on transition)
  - Open Hand                   -> idle (like lifting off the trackpad)

Controls:
  - Press 'q' to quit.
  - Move mouse to top-left corner to trigger pyautogui failsafe.
"""

import math
import os

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
    drawing_styles,
)
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

BaseOptions = mp.tasks.BaseOptions

# ── Landmark indices for finger tips and PIP joints ──────────────
FINGER_TIPS = [
    HandLandmark.INDEX_FINGER_TIP,
    HandLandmark.MIDDLE_FINGER_TIP,
    HandLandmark.RING_FINGER_TIP,
    HandLandmark.PINKY_TIP,
]
FINGER_PIPS = [
    HandLandmark.INDEX_FINGER_PIP,
    HandLandmark.MIDDLE_FINGER_PIP,
    HandLandmark.RING_FINGER_PIP,
    HandLandmark.PINKY_PIP,
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ── Tunable constants ────────────────────────────────────────────
PINCH_THRESHOLD = 0.04       # normalised distance to count as a pinch
RIGHT_PINCH_THRESHOLD = 0.035 # tighter threshold for right pinch to avoid false triggers
SENSITIVITY = 2000            # multiplier for relative cursor movement
SCROLL_SENSITIVITY = 60       # multiplier for scroll deltas
SMOOTHING_ALPHA = 0.4         # exponential smoothing (0 = sluggish, 1 = raw)
GESTURE_CONFIRM_FRAMES = {
    "Pinch": 1,
    "Pointing": 1,
    "Right Pinch": 2,
    "default": 3,
}
DEAD_ZONE = 0.002             # ignore raw movement deltas smaller than this

# ── pyautogui setup ──────────────────────────────────────────────
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0           # no artificial delay between calls


# ── Helpers ──────────────────────────────────────────────────────
def _dist(a, b) -> float:
    """Euclidean distance between two landmarks (normalised coords)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def classify_gesture(landmarks) -> str:
    """Return a gesture label, including pinch-based gestures."""
    lm = landmarks

    thumb_tip = lm[HandLandmark.THUMB_TIP]
    thumb_ip = lm[HandLandmark.THUMB_IP]
    index_tip = lm[HandLandmark.INDEX_FINGER_TIP]
    middle_tip = lm[HandLandmark.MIDDLE_FINGER_TIP]

    # Pinch detection (distance-based, checked first)
    if _dist(thumb_tip, index_tip) < PINCH_THRESHOLD:
        return "Pinch"
    if _dist(thumb_tip, middle_tip) < RIGHT_PINCH_THRESHOLD:
        return "Right Pinch"

    thumb_up = thumb_tip.x < thumb_ip.x  # mirrored webcam

    fingers_up = [lm[tip].y < lm[pip].y for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]
    total_up = sum(fingers_up) + int(thumb_up)

    if total_up == 0:
        return "Fist"
    if total_up == 5:
        return "Open Hand"
    if fingers_up == [True, False, False, False]:
        return "Pointing"
    if fingers_up == [True, True, False, False]:
        return "Peace / Victory"
    if fingers_up == [False, False, False, True] and not thumb_up:
        return "Pinky Up"
    if thumb_up and sum(fingers_up) == 0:
        return "Thumbs Up"
    if fingers_up == [True, False, False, True] and not thumb_up:
        return "Rock On"

    return "Idle"

def get_screen_center():
    screen_width, screen_height = pyautogui.size()
    center_x = screen_width // 2
    center_y = screen_height // 2
    return center_x, center_y

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found at {MODEL_PATH}")
        print("Download it with:")
        print("  curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    landmarker = HandLandmarker.create_from_options(options)
    hand_connections = HandLandmarksConnections.HAND_CONNECTIONS
    frame_timestamp_ms = 0

    # ── State tracking ───────────────────────────────────────────
    prev_gesture = "Idle"
    prev_pip_pos = None      # previous index-finger PIP (normalised x, y)
    prev_tip = None          # previous index-finger-tip for scroll deltas
    smooth_dx = 0.0          # smoothed delta x
    smooth_dy = 0.0          # smoothed delta y
    dragging = False         # currently holding left button?
    confirmed_gesture = "Idle"
    gesture_candidate = "Idle"
    candidate_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        gesture = "Idle"
        pip_pos = None
        tip_pos = None

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # Draw skeleton on preview
            drawing_utils.draw_landmarks(
                frame,
                landmarks,
                hand_connections,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

            gesture = classify_gesture(landmarks)

            # PIP joint for stable cursor tracking (barely moves during pinch)
            idx_pip = landmarks[HandLandmark.INDEX_FINGER_PIP]
            pip_pos = (idx_pip.x, idx_pip.y)

            # Fingertip position still used for scroll deltas
            idx_tip = landmarks[HandLandmark.INDEX_FINGER_TIP]
            tip_pos = (idx_tip.x, idx_tip.y)

            # Determine handedness for the label
            hand_label = "Hand"
            if result.handedness:
                hand_label = result.handedness[0][0].category_name

            wrist = landmarks[HandLandmark.WRIST]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(
                frame,
                f"{hand_label}: {confirmed_gesture}",
                (cx - 60, cy - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # ── Gesture hysteresis: require N consecutive frames to switch ─
        if gesture != gesture_candidate:
            gesture_candidate = gesture
            candidate_count = 1
        else:
            candidate_count += 1

        required = GESTURE_CONFIRM_FRAMES.get(gesture_candidate, GESTURE_CONFIRM_FRAMES["default"])
        if candidate_count >= required:
            confirmed_gesture = gesture

        # ── State machine: act on confirmed gesture ───────────────
        just_entered = confirmed_gesture != prev_gesture

        if confirmed_gesture == "Pointing":
            if dragging:
                pyautogui.mouseUp(button="left")
                dragging = False

            if prev_pip_pos is not None and pip_pos is not None:
                raw_dx = (pip_pos[0] - prev_pip_pos[0]) * SENSITIVITY
                raw_dy = (pip_pos[1] - prev_pip_pos[1]) * SENSITIVITY
                if abs(raw_dx) < DEAD_ZONE * SENSITIVITY and abs(raw_dy) < DEAD_ZONE * SENSITIVITY:
                    raw_dx = 0.0
                    raw_dy = 0.0
                smooth_dx = SMOOTHING_ALPHA * raw_dx + (1 - SMOOTHING_ALPHA) * smooth_dx
                smooth_dy = SMOOTHING_ALPHA * raw_dy + (1 - SMOOTHING_ALPHA) * smooth_dy
                pyautogui.moveRel(int(smooth_dx), int(smooth_dy), _pause=False)

        elif confirmed_gesture == "Pinch":
            if just_entered:
                pyautogui.mouseDown(button="left")
                dragging = True
            elif dragging and prev_pip_pos is not None and pip_pos is not None:
                raw_dx = (pip_pos[0] - prev_pip_pos[0]) * SENSITIVITY
                raw_dy = (pip_pos[1] - prev_pip_pos[1]) * SENSITIVITY
                if abs(raw_dx) < DEAD_ZONE * SENSITIVITY and abs(raw_dy) < DEAD_ZONE * SENSITIVITY:
                    raw_dx = 0.0
                    raw_dy = 0.0
                smooth_dx = SMOOTHING_ALPHA * raw_dx + (1 - SMOOTHING_ALPHA) * smooth_dx
                smooth_dy = SMOOTHING_ALPHA * raw_dy + (1 - SMOOTHING_ALPHA) * smooth_dy
                pyautogui.moveRel(int(smooth_dx), int(smooth_dy), _pause=False)

        elif confirmed_gesture == "Right Pinch":
            if just_entered:
                pyautogui.click(button="right", _pause=False)

        elif confirmed_gesture == "Peace / Victory":
            if prev_tip is not None and tip_pos is not None:
                scroll_delta = (prev_tip[1] - tip_pos[1]) * SCROLL_SENSITIVITY
                if abs(scroll_delta) > 0.3:
                    pyautogui.scroll(int(scroll_delta), _pause=False)

        elif confirmed_gesture == "Fist":
            if just_entered:
                if dragging:
                    pyautogui.mouseUp(button="left")
                    dragging = False
                pyautogui.doubleClick(_pause=False)

        elif confirmed_gesture == "Rock On":
            pyautogui.moveTo(get_screen_center())

        else:
            if dragging:
                pyautogui.mouseUp(button="left")
                dragging = False
            smooth_dx = 0.0
            smooth_dy = 0.0

        prev_gesture = confirmed_gesture
        prev_pip_pos = pip_pos
        prev_tip = tip_pos

        cv2.imshow("Hand Gesture Mouse Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if dragging:
        pyautogui.mouseUp(button="left")
 
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
