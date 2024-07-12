# Lingclusive Project's Backend Development

This project is a Universal Handsign Translator that uses Mediapipe and PyTorch to recognize hand gestures. The system collects data from images or real-time video, preprocesses the data, trains a neural network model, and predicts gestures from live video input. The project consists of several scripts for data collection, preprocessing, model training, and prediction.

## Setup Instructions
### Prerequisites
- Flask         == 3.0.3
- Flask_cors    == 4.0.1
- Mediapipe     == 0.10.14
- Numpy         == 2.0.0
- Pandas        == 2.2.2
- Pillow        == 10.4.0
- Scikit_learn  == 1.5.1
- Torch         == 2.3.1

## Installation

1. Clone the repository:

```bash
  git clone https://github.com/discuss-gh5/backend.git
  git clone https://github.com/discuss-gh5/frontend.git
```
    
2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:

Rename ```env``` to ```.env``` and update the paths accordingly:
```bash
MODEL_ADDRESS           = model
IMAGE_ADDRESS           = images
CURRENT_KEYPOINT_NAME   = all_keypoint
```
## Run Locally

Assuming you have cloned the repo, you can run the Flask app to use the gesture recognition system via a web interface. Start the app by running:

```bash
  python app.py
```
