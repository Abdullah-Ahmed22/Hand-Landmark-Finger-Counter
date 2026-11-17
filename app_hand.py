import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CAMERA_WIDTH = 2000
CAMERA_HEIGHT = 1500
CAMERA_INDEX = 0

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def is_finger_open(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y

def count_fig_lift(landmarks):
    list_ = []
    finger_pairs = [
        (8, 6),    # Index
        (12, 10),  # Middle
        (16, 14),  # Ring
        (20, 18)   # Pinky
    ]
    for tip, pip in finger_pairs:
        list_.append(is_finger_open(landmarks, tip, pip))

    if landmarks[5].x > landmarks[17].x:
        list_.append(landmarks[4].x > landmarks[3].x)
    else:
        list_.append(landmarks[4].x < landmarks[3].x)

    return sum(list_)


def count_fig_right(landmarks):
    list_ = []
    finger_pairs = [
        (8, 6),    # Index
        (12, 10),  # Middle
        (16, 14),  # Ring
        (20, 18)   # Pinky
    ]
    for tip, pip in finger_pairs:
        list_.append(is_finger_open(landmarks, tip, pip))

    if landmarks[4].x < landmarks[17].x:
        list_.append(landmarks[4].x < landmarks[3].x)
    else:
        list_.append(landmarks[4].x > landmarks[3].x)

    return sum(list_)


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# STEP: Create an ObjectDetector object
print("Loading Model...")
base_options = python.BaseOptions(model_asset_path='Models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
print("Model loaded successfuly!")

print("Starting webcam...")
cap = cv2.VideoCapture(CAMERA_INDEX)
time.sleep(2.0) 

print("Webcam started!")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame from webcam.")
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect(mp_image)
    annotated_image = draw_landmarks_on_image(image.copy(), detection_result)

    total_left = 0
    total_right = 0
    if detection_result.hand_landmarks:


        for i, landmarks in enumerate(detection_result.hand_landmarks):
            hand_label = ( detection_result.handedness[i][0].category_name)

            if hand_label == "Left":
                total_left = count_fig_lift(landmarks)
            else:
                total_right = count_fig_right(landmarks)

    cv2.putText(
        annotated_image,
        f"Left Hand: {total_left}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_image,
        f"Right Hand: {total_right}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_image,
        f"Total Fingers: {total_left + total_right}",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )



    cv2.putText(annotated_image, "Press 'q' to quit", (30,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
detector.close()

print("Webcam released and allwindows closed.")
print("Done program.")
