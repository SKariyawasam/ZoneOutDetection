import cv2
import time

def start_webcam_stream():
    """Captures real-time video feed from the built-in webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # In a real setup, we'd pass the frame to the feature extractor here
        cv2.imshow('Visual Processing Node (Module A)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_webcam_stream()
