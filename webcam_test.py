import cv2

CAMERA_INDEX = 0  # 0 = first camera; try 1 if you have several

# CAP_DSHOW is a Windows camera backend that usually opens faster
# and more reliably than the default one.
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise SystemExit("Could not open the camera. Check permissions or other apps using it.")

print("Camera opened. Press 'q' in the video window to quit.")

try:
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("Failed to read a frame from the camera.")
            break

        cv2.imshow("Webcam test", frame)

       
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released.")