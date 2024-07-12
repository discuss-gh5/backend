from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
from PIL import Image
from init import model_address

app = Flask(__name__)

df = pd.read_csv(f"{model_address}/keypoint.csv")
label_encoder = LabelEncoder()
label_encoder.fit(df["label"])


class DiscussModel(nn.Module):
    def __init__(self):
        super(DiscussModel, self).__init__()
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


model = DiscussModel()
model.load_state_dict(torch.load(f"{model_address}/lingclusive_model.pth"))
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


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


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(BytesIO(file.read()))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        predictions = []
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            row = [landmark.x for landmark in landmarks]
            row += [landmark.y for landmark in landmarks]
            row += [landmark.z for landmark in landmarks]

            input_tensor = torch.tensor([row], dtype=torch.float32)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                predictions.append(predicted.numpy()[0])

        gesture = get_gesture_name(predictions, label_encoder)
    else:
        gesture = "undefined"

    return jsonify({"gesture": gesture})


if __name__ == "__main__":
    app.run(debug=True)
