import cv2
import time
import glob
import numpy as np
import os

def benchmark():
    img_dir = r"C:\Users\User\Desktop\FYP\Prototype\Datasets\DipSeer Dataset\subject_01\images"
    if not os.path.exists(img_dir):
        img_dir = r"C:\Users\User\Desktop\FYP\Datasets\DipSeer Dataset\subject_01\images"
        
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:100]
    if not img_files:
        print("No test images found for benchmark.")
        return
        
    frames = [cv2.imread(f) for f in img_files if cv2.imread(f) is not None]
    print(f"Loaded {len(frames)} frames ({frames[0].shape}) for CPU latency benchmark.\n")
    
    # 1. Haar Cascade Face Only
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    t0 = time.time()
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    t_haar_face = (time.time() - t0) / len(frames) * 1000.0
    
    # 2. Haar Cascade Face + Eye Cascade (Anh et al., 2024 region)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    t0 = time.time()
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi_gray = gray[y:y+int(h*0.6), x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
    t_haar_eye = (time.time() - t0) / len(frames) * 1000.0
    
    # 3. Haar Cascade with 15-frame Bounding Box Caching (Our Prototype Implementation)
    t0 = time.time()
    cached_box = None
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cached_box is None or idx % 15 == 0:
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)
            if len(faces) > 0:
                cached_box = faces[0]
    t_haar_cached = (time.time() - t0) / len(frames) * 1000.0
    
    # 4. MediaPipe Face Detection (if installed)
    mp_latency = None
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        t0 = time.time()
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_face.process(rgb)
        mp_latency = (time.time() - t0) / len(frames) * 1000.0
    except Exception as e:
        mp_latency = "N/A"
        
    print("================================================================")
    print("DETECTOR EXECUTION LATENCY BENCHMARK ON CPU (A-20):")
    print("================================================================")
    print(f"1. Haar Face Detection (Full Frame every step) : {t_haar_face:.2f} ms/frame ({1000.0/t_haar_face:.1f} FPS)")
    print(f"2. Haar Face + Eye Cascade (Anh et al., 2024)  : {t_haar_eye:.2f} ms/frame ({1000.0/t_haar_eye:.1f} FPS)")
    print(f"3. Prototype Cached Haar Cascade (every 15 f)  : {t_haar_cached:.2f} ms/frame ({1000.0/t_haar_cached:.1f} FPS)")
    if isinstance(mp_latency, float):
        print(f"4. MediaPipe Face Detection (BlazeFace CPU)    : {mp_latency:.2f} ms/frame ({1000.0/mp_latency:.1f} FPS)")
    print("================================================================")

if __name__ == '__main__':
    benchmark()
