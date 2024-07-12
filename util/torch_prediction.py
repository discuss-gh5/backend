import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from init import model_address

import numpy as np
import torch
import cv2
import mediapipe as mp
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(f"{model_address}/all_keypoint.csv")
label_encoder = LabelEncoder()
label_encoder.fit(df["label"])


class Lingclusive(nn.Module):
    def __init__(self):
        super(Lingclusive, self).__init__()
        self.fc1 = nn.Linear(df.shape[1] - 1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(label_encoder.classes_))
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


model = Lingclusive()
model.load_state_dict(torch.load(f"{model_address}/lingclusive_model.pth"))
model.eval()


def get_gesture_name(predictions, label_encoder):
    if len(predictions) == 2:
        if predictions[0] == predictions[1]:
            return label_encoder.inverse_transform([predictions[0]])[0]
        else:
            return "undefined"
    elif len(predictions) == 1:
        return label_encoder.inverse_transform([predictions[0]])[0]
    else:
        return "undefined"


def predict_gestures(model, input_data, label_encoder):
    with torch.no_grad():
        inputs = torch.tensor(input_data, dtype=torch.float32)
        outputs = model(inputs)
        _, predictions = torch.max(outputs.data, 1)
        return get_gesture_name(predictions, label_encoder)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            row = [landmark.x for landmark in landmarks]
            row += [landmark.y for landmark in landmarks]
            row += [landmark.z for landmark in landmarks]

            input_tensor = torch.tensor([row], dtype=torch.float32)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                sign = label_encoder.inverse_transform(predicted.numpy())[0]

            cv2.putText(
                frame,
                sign,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Universal Handsign Translator", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
