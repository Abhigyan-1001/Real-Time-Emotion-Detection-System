import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.predictor import EmotionPredictor
from src.preprocessing import to_model_input

ZIP_PATH = Path.home() / "Downloads" / "archive.zip"  # edit if your zip is elsewhere
PER_CLASS = 20

predictor = EmotionPredictor("models/emotion_cnn_v1.keras", "models/labels.json")

by_class = defaultdict(list)
with zipfile.ZipFile(ZIP_PATH) as zf:
    for name in sorted(zf.namelist()):
        parts = name.split("/")
        is_image = name.lower().endswith((".jpg", ".jpeg", ".png"))
        if len(parts) == 3 and parts[0] == "train" and is_image:
            if len(by_class[parts[1]]) < PER_CLASS:
                by_class[parts[1]].append(name)

    total_ok = total = 0
    print(f"{'class':<10} correct / checked")
    for cls in predictor.labels:
        ok = 0
        names = by_class[cls]
        for name in names:
            data = np.frombuffer(zf.read(name), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            model_input = to_model_input(image)
            if model_input is None:
                continue
            label, _ = predictor.top_label(predictor.predict_scores(model_input)[0])
            ok += label == cls
        print(f"{cls:<10} {ok:>3} / {len(names)}")
        total_ok += ok
        total += len(names)
    print(f"{'overall':<10} {total_ok:>3} / {total}")
    