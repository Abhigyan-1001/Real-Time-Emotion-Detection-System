import json
from pathlib import Path

import keras
import numpy as np

MODEL_PATH = Path("models/emotion_cnn_v1.keras")
LABELS_PATH = Path("models/labels.json")

model = keras.models.load_model(MODEL_PATH)
labels = json.loads(LABELS_PATH.read_text())

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)
print("Parameters  :", f"{model.count_params():,}")
print("Labels      :", dict(enumerate(labels)))

# A blank 48x48 grayscale image, pixel values 0-255 (the model rescales internally).
dummy = np.zeros((1, 48, 48, 1), dtype="float32")
probs = model.predict(dummy, verbose=0)[0]
print("Probabilities sum to:", round(float(probs.sum()), 4))