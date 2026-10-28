import time

import cv2

from src.face_detector import FaceDetector

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

detector = FaceDetector(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Could not open the camera.")

print("Press 'q' in the video window to quit.")

try:
    while True:
        start = time.perf_counter()
        success, frame = cap.read()
        if not success or frame is None:
            print("Failed to read a frame.")
            break

        faces = detector.detect(frame)

        for face in faces:
            cv2.rectangle(frame, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)
            cv2.putText(frame, f"{face.score:.2f}", (face.x, max(face.y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for px, py in face.landmarks:
                cv2.circle(frame, (int(px), int(py)), 3, (0, 0, 255), -1)

        fps = 1.0 / (time.perf_counter() - start)
        status = f"Faces: {len(faces)} | ~{fps:.0f} FPS"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YuNet face detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()