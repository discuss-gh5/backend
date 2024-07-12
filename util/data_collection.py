import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from init import model_address
import csv
import cv2
import mediapipe as mp
import time


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

keypoint_file = open(f"{model_address}/keypoint.csv", "w", newline="")
csv_writer = csv.writer(keypoint_file)
csv_writer.writerow(
    ["label"]
    + [f"x{i}" for i in range(21)]
    + [f"y{i}" for i in range(21)]
    + [f"z{i}" for i in range(21)]
)


def collect_data_from_image(image_path, label):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            row = [label]
            row += [landmark.x for landmark in landmarks]
            row += [landmark.y for landmark in landmarks]
            row += [landmark.z for landmark in landmarks]
            print(label)
            csv_writer.writerow(row)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Landmarks", image)
    cv2.destroyAllWindows()


def collect_data_manually(label):
    print(f"Scanning: {label}")
    input("Press enter if ready")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                row = [label]
                row += [landmark.x for landmark in landmarks]
                row += [landmark.y for landmark in landmarks]
                row += [landmark.z for landmark in landmarks]
                csv_writer.writerow(row)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Collect Data", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            print("Next up...")
            break


# skip_labels = ["J", "Z"]
# labels = [
#     chr(i) for i in range(ord("A"), ord("Z") + 1) if chr(i).upper() not in skip_labels
# ]
# for label in labels:
#     collect_data_from_image(f"images/{label}.png", label)

labels = ["One", "Two", "Three", "Four", "Five"]
for label in labels:
    collect_data_manually(label)
