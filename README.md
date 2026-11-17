# Hand Landmark Finger Counter

A real-time webcam demo that detects up to two hands, draws their landmarks, and counts the raised fingers on each hand using MediaPipe and OpenCV.

## Features

- Detects up to two hands in the webcam image.
- Displays hand landmarks and the detected handedness.
- Shows raised-finger counts for the left hand, right hand, and both hands combined.
- Press **Q** in the video window to exit.

## Requirements

- Python 3.9 or newer
- A working webcam
- A graphical desktop session for the OpenCV video window

The MediaPipe model is included at `Models/hand_landmarker.task`.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

## Run

From the project directory, run:

```bash
python app_hand.py
```

Allow webcam access if prompted. The annotated webcam feed opens in an OpenCV window. Press **Q** while that window is focused to close it.

The camera index and requested capture resolution are set near the top of `app_hand.py` (`CAMERA_INDEX`, `CAMERA_WIDTH`, and `CAMERA_HEIGHT`). Change these values if your camera uses a different index or resolution.

## Notes

- The finger count is based on landmark positions and may be less reliable for unusual hand angles, occlusion, or mirrored camera feeds.
- Although `requirements.txt` includes Streamlit and other packages, the current application is a desktop OpenCV script; it is not launched with `streamlit run`.
