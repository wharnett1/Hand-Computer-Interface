# Hand Gesture Mouse Control

**WORK IN PROGRESS**

Control your mouse cursor with hand gestures using a webcam, powered by MediaPipe hand tracking.

## Gestures

| Gesture | Action |
|---|---|
| Point (index finger only) | Move cursor (relative, like a trackpad) |
| Thumb + index pinch | Left click / click-and-drag |
| Thumb + middle finger pinch | Right click |
| Peace / Victory sign | Scroll up/down |
| Fist | Double click |
| Rock On (index + pinky) | Snap cursor to screen center |
| Open Hand | Idle (like lifting off a trackpad) |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the hand landmark model

```bash
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 3. Run

```bash
python hand_detect_pip.py
```

Press **q** to quit. Move the mouse to the top-left corner of the screen to trigger the pyautogui failsafe.

## Tuning

Constants at the top of the script control feel and responsiveness:

| Constant | Default | Description |
|---|---|---|
| `PINCH_THRESHOLD` | `0.04` | Normalised distance for left pinch detection |
| `RIGHT_PINCH_THRESHOLD` | `0.035` | Tighter threshold for right pinch (reduces false triggers) |
| `SENSITIVITY` | `2000` | Multiplier for relative cursor movement |
| `SCROLL_SENSITIVITY` | `60` | Multiplier for scroll speed |
| `SMOOTHING_ALPHA` | `0.4` | Exponential smoothing factor (0 = sluggish, 1 = raw) |
| `DEAD_ZONE` | `0.002` | Ignores raw movement deltas smaller than this |
| `GESTURE_CONFIRM_FRAMES` | per-gesture | Frames required to confirm a gesture switch (prevents flicker) |

## How it works

- **PIP joint tracking** — Cursor movement tracks the index finger's middle knuckle (PIP joint) instead of the fingertip. The PIP barely moves when you close your fingertip to your thumb, eliminating the cursor drift that normally happens right before a pinch click.
- **Gesture hysteresis** — A new gesture must be detected for N consecutive frames before the system acts on it. Pinch and Pointing confirm in 1 frame (instant clicks), while other gestures use a 3-frame buffer to prevent flicker.
- **Dead zone** — Micro-jitters below the dead zone threshold are zeroed out before smoothing, keeping the cursor still when your hand is stationary.

## Requirements

- Python 3.10+
- Webcam
- macOS (pyautogui mouse control; Linux/Windows should also work but are untested)
