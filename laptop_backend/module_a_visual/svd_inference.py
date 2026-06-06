import cv2
import numpy as np
import threading
import tsfel
import pickle
import pandas as pd
from collections import deque
import time

class VisualInferenceModel:
    def __init__(self, model_path=r"c:\Users\User\Desktop\FYP\offline_training\svd_xgboost_model.pkl",
                 features_path=r"c:\Users\User\Desktop\FYP\offline_training\svd_feature_cols.pkl"):
        print("Loading SVD XGBoost Model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            self.model.set_params(device="cpu") # Run inference on CPU
            
        with open(features_path, 'rb') as f:
            self.feature_cols = pickle.load(f)
            
        self.cfg = tsfel.get_features_by_domain()
        if 'temporal' in self.cfg:
            del self.cfg['temporal']
            
        # Disable GPU for live inference to avoid CudnnRNN conflicts with the physio LSTM model
        print("Loading OpenCV Cascades (Frontal + Profile)...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        print("Opening webcam...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("Webcam opened:", self.cap.isOpened())
        
        # 10 seconds of data at 20 FPS = 200 frames.
        self.window_size = 200 
        self.energy_buffer = deque(maxlen=self.window_size)
        
        self.running = True
        self.face_detected = False
        
        # Start background capture thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("SVD Visual Inference Initialized.")

    def _capture_loop(self):
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            frame_count += 1
            if frame_count % 3 == 0:
                continue # Skip 1 in 3 frames for 20 FPS
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Try frontal face first (alt2 is more robust to occlusions)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
            
            # If no frontal face, try profile face
            if len(faces) == 0:
                faces = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
                # Also try flipping the image for the other profile
                if len(faces) == 0:
                    flipped_gray = cv2.flip(gray, 1)
                    flipped_faces = self.profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.2, minNeighbors=4)
                    if len(flipped_faces) > 0:
                        # Convert coordinates back
                        x, y, w, h = flipped_faces[0]
                        faces = [[gray.shape[1] - x - w, y, w, h]]
            
            if len(faces) == 0:
                self.face_detected = False
                self.energy_buffer.append(np.zeros(64))
                continue
                
            self.face_detected = True
            
            # Use largest face
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            
            # The eyes are roughly in the top half of the face
            y_eye_top = int(y + h * 0.15)
            y_eye_bottom = int(y + h * 0.55)
            x_eye_left = int(x + w * 0.1)
            x_eye_right = int(x + w * 0.9)
            
            eye_region = gray[y_eye_top:y_eye_bottom, x_eye_left:x_eye_right]
            if eye_region.shape[0] == 0 or eye_region.shape[1] == 0:
                self.energy_buffer.append(np.zeros(64))
                continue
                
            eye_crop = cv2.resize(eye_region, (128, 96))
            
            block_energies = []
            for row in range(8):
                for col in range(8):
                    block = eye_crop[row*12:(row+1)*12, col*16:(col+1)*16]
                    block_energies.append(np.sum(block.astype(np.float32) ** 2))
                    
            self.energy_buffer.append(np.array(block_energies))

    def predict(self):
        if not self.face_detected or len(self.energy_buffer) < 30: # Need at least 1.5s of data
            return -1.0
            
        E = np.array(self.energy_buffer)
        
        # If matrix is empty or all zeros
        if np.all(E == 0):
            return -1.0
            
        # Use numpy SVD — same result as tf.linalg.svd, no GPU dependency
        U_np, S, Vt = np.linalg.svd(E.astype(np.float32), full_matrices=False)
        
        best_vector = None
        max_amplitude = -1
        
        # Find highest high-frequency amplitude among top singular vectors
        for i in range(1, min(U_np.shape[1], 10)):
            u_i = U_np[:, i]
            denom = (np.max(u_i) - np.min(u_i) + 1e-6)
            u_i = (u_i - np.min(u_i)) / denom
            
            fft_vals = np.abs(np.fft.rfft(u_i))
            hf_amp = np.max(fft_vals[len(fft_vals)//2:])
            if hf_amp > max_amplitude:
                max_amplitude = hf_amp
                best_vector = u_i
                
        if best_vector is None:
            best_vector = U_np[:, 0]
            
        features = tsfel.time_series_features_extractor(self.cfg, best_vector, fs=20, verbose=0)
        
        # Align columns
        for col in self.feature_cols:
            if col not in features.columns:
                features[col] = 0.0
                
        X = features[self.feature_cols]
        probs = self.model.predict_proba(X)
        
        # Predict engagement/focus
        raw_prob = float(probs[0][1])
        
        # Calibrate overly-confident XGBoost predictions for live webcam data
        # A power of 8.0 is a balanced middle ground: 0.98 -> 0.85, 0.95 -> 0.66, 0.90 -> 0.43
        prob_focused = raw_prob ** 8.0
        
        return prob_focused

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
