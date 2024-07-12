import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from init import model_address

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

keypoint_name = "all_keypoint"

df = pd.read_csv(f"{model_address}/{keypoint_name}.csv")

X = df.drop("label", axis=1)
y = df["label"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(f"Unique labels after encoding: {sorted(set(y))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Unique labels in y_train: {sorted(set(y_train))}")
print(f"Unique labels in y_test: {sorted(set(y_test))}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

X_train.to_csv(f"{model_address}/X_train.csv", index=False)
X_test.to_csv(f"{model_address}/X_test.csv", index=False)
pd.Series(y_train).to_csv(f"{model_address}/y_train.csv", index=False, header=False)
pd.Series(y_test).to_csv(f"{model_address}/y_test.csv", index=False, header=False)


def get_train_test_split():
    return X_train, y_train, X_test, y_test
