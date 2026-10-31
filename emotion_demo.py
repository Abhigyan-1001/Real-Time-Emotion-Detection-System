import time

import cv2

from src.face_detector import FaceDetector
from src.pipeline import analyze_frame
from src.predictor import EmotionPredictor

detector = FaceDetector("models/face_detection_yunet_2023mar.onnx")
predictor = EmotionPredictor("models/emotion_cnn_v1.keras", "models/labels.json")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Could not open the camera.")

print("Press 'q' in the video window to quit. Nothing is saved or recorded.")

try:
    while True:
        start = time.perf_counter()
        success, frame = cap.read()
        if not success or frame is None:
            print("Failed to read a frame.")
            break

        results = analyze_frame(frame, detector, predictor)

        for r in results:
            d = r.detection
            cv2.rectangle(frame, (d.x, d.y), (d.x + d.w, d.y + d.h), (0, 255, 0), 2)
            text = f"{r.label} {r.score:.0%}"
            cv2.putText(frame, text, (d.x, max(d.y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        fps = 1.0 / (time.perf_counter() - start)
        status = f"Faces: {len(results)} | ~{fps:.0f} FPS" if results else f"No face detected | ~{fps:.0f} FPS"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Model output = visual expression pattern, not true emotion",
                    (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Emotion demo (raw predictions)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()