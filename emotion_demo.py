import time

import cv2

from src.face_detector import FaceDetector
from src.pipeline import analyze_frame
from src.postprocessing import FaceTracker
from src.predictor import EmotionPredictor

ALPHA = 0.15      # smoothing strength: 1.0 = off, smaller = smoother but slower to react
THRESHOLD = 0.40  # tentative; we will calibrate it from validation data later

detector = FaceDetector("models/face_detection_yunet_2023mar.onnx")
predictor = EmotionPredictor("models/emotion_cnn_v1.keras", "models/labels.json")
tracker = FaceTracker(predictor.labels, alpha=ALPHA, threshold=THRESHOLD)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Could not open the camera.")

print("Keys: 's' = toggle smoothing, 'q' = quit. Nothing is saved or recorded.")
smoothing_on = True

try:
    while True:
        start = time.perf_counter()
        success, frame = cap.read()
        if not success or frame is None:
            print("Failed to read a frame.")
            break

        # The tracker must see every frame (even with no faces) so it can age old tracks.
        faces = tracker.update(analyze_frame(frame, detector, predictor))

        for f in faces:
            d = f.detection
            if smoothing_on:
                label, score = f.label, f.score
                color = (0, 215, 255) if f.uncertain else (0, 255, 0)  # yellow if uncertain
            else:
                label, score, color = f.raw_label, f.raw_score, (255, 160, 0)
            cv2.rectangle(frame, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
            cv2.putText(frame, f"#{f.track_id} {label} {score:.0%}", (d.x, max(d.y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps = 1.0 / (time.perf_counter() - start)
        mode = f"smoothed (alpha={ALPHA})" if smoothing_on else "RAW"
        status = f"Faces: {len(faces)}" if faces else "No face detected"
        cv2.putText(frame, f"{status} | {mode} | ~{fps:.0f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Model output = visual expression pattern, not true emotion",
                    (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Emotion demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            smoothing_on = not smoothing_on
finally:
    cap.release()
    cv2.destroyAllWindows()
