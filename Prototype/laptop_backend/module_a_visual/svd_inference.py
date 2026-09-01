import os
import cv2
import numpy as np
import threading
import tsfel
import pickle
import pandas as pd
from collections import deque
import time

class VisualInferenceModel:
    def __init__(self, model_path=None, features_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        proto_offline = os.path.abspath(os.path.join(base_dir, "..", "..", "offline_training"))
        
        if model_path is None:
            local_m = os.path.join(base_dir, "svd_xgboost_model.pkl")
            off_m = os.path.join(proto_offline, "svd_xgboost_model.pkl")
            model_path = local_m if os.path.exists(local_m) else off_m
            
        if features_path is None:
            local_f = os.path.join(base_dir, "svd_feature_cols.pkl")
            off_f = os.path.join(proto_offline, "svd_feature_cols.pkl")
            features_path = local_f if os.path.exists(local_f) else off_f

        print(f"Loading SVD XGBoost Model from {model_path}...")
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
        # Robust camera opening across directshow, msmf, and default backends
        print("Opening webcam...")
        self.cap = None
        for idx in [0, 1]:
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                c = cv2.VideoCapture(idx, backend)
                if c.isOpened():
                    ret, test_frame = c.read()
                    if ret and test_frame is not None:
                        self.cap = c
                        break
                    else:
                        c.release()
            if self.cap is not None:
                break
                
        if self.cap is None:
            print("[WARN] No functional camera opened. Creating fallback capture.")
            self.cap = cv2.VideoCapture(0)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Read back driver-negotiated hardware resolution
        self.actual_w = float(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640.0
        self.actual_h = float(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480.0
        self.res_scale = self.actual_w / 640.0
        print(f"Webcam opened successfully (Requested: 640x480, Actual: {int(self.actual_w)}x{int(self.actual_h)}, Scale: {self.res_scale:.2f}):", self.cap.isOpened())
        
        # 10 seconds of data at 20 FPS = 200 frames.
        self.window_size = 200 
        self.energy_buffer = deque(maxlen=self.window_size)
        self.head_pose_buffer = deque(maxlen=self.window_size)
        
        self.running = True
        self.face_detected = False
        
        # Start background capture thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("SVD Visual Inference Initialized with Bosch & D'Mello (2021) Dynamics.")

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
            
            # Record head pose dynamics (Bosch & D'Mello, 2021)
            self.head_pose_buffer.append(np.array([x + w/2.0, y + h/2.0, float(w), float(h)], dtype=np.float32))
            
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
        
        # Predict engagement/focus probability directly from balanced XGBoost model
        prob_focused = float(probs[0][1])
        return prob_focused

    def _calculate_head_dynamics(self):
        """
        Extracts temporal head pose velocity and jitter (Bosch & D'Mello, 2021).
        Differentiates natural active engagement from fatigue/head dropping.
        Velocity is normalized by face width (W_face/s), making it completely
        distance-independent and resolution-independent.
        """
        if len(self.head_pose_buffer) < 20:
            return 0.80
            
        pose_arr = np.array(self.head_pose_buffer)
        centroids = pose_arr[:, :2]
        face_w = np.median(pose_arr[:, 2]) + 1e-6
        
        # 1st-derivative velocity of head centroid in pixels/frame
        displacements = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        
        # Convert to distance-independent velocity in face-widths per second (at 20 FPS)
        velocities_fw_per_sec = (displacements / face_w) * 20.0
        mean_vel = np.mean(velocities_fw_per_sec)
        jitter = np.std(velocities_fw_per_sec)
        
        # Normal active micro-movement: 0.02 - 0.15 W_face/s
        # Extreme jitter (>0.25 W_face/s) or complete freeze (<0.005 W_face/s) indicates disengagement
        vel_ceiling = 0.25 # W_face/s
        vel_floor = 0.005  # W_face/s
        jitter_denom = 0.15 # W_face/s
        
        if mean_vel > vel_ceiling:
            stability = max(0.2, 1.0 - (mean_vel / (2.0 * vel_ceiling)))
        elif mean_vel < vel_floor:
            stability = 0.50 # Unnaturally stiff/frozen
        else:
            stability = float(np.clip(1.0 - (jitter / jitter_denom), 0.5, 0.95))
            
        return float(stability)

    def _calculate_blank_stare_index(self):
        """
        Detects prolonged saccadic stillness (the 'Blank Stare' indicator).
        Mind-wandering often presents as frozen gaze with near-zero micro-saccades.
        Employs a continuous, piecewise-linear transfer function invariant to global
        multiplicative illumination scaling.
        """
        if len(self.energy_buffer) < 60:
            return 0.75
            
        recent_E = np.array(list(self.energy_buffer)[-60:]) # Last 3 seconds (60 frames at 20fps)
        if np.all(recent_E == 0):
            return 0.50
            
        # Invariant to global multiplicative changes in illumination:
        # P -> alpha * P  =>  E -> alpha^2 * E  =>  std_t(E) -> alpha^2 * std_t(E)
        # Ratio cancels alpha^2 exactly (epsilon = 1e-6 acts as numerical division guard):
        mean_energy = np.mean(recent_E) + 1e-6
        relative_saccade_variation = float(np.mean(np.std(recent_E, axis=0)) / mean_energy)
        
        # Continuous piecewise-linear transfer function without step-discontinuities:
        # - r_low = 0.030: One-sided lower bound set below 5th percentile of active focus (0.046) - blank stare plateau [score=0.30]
        # - r in [0.03, 0.12]: Smooth linear transition [0.30, 0.95]
        # - r_high = 0.120: 75th percentile of active exploration across focused recordings - active gaze saturation [score=0.95]
        r_low, r_high = 0.03, 0.12
        blank_stare_score = float(np.interp(relative_saccade_variation, [r_low, r_high], [0.30, 0.95]))
            
        return blank_stare_score

    def get_image_traits(self):
        """
        Extracts 4 key spatial image traits enriched with Bosch & D'Mello (2021) dynamics:
        [gaze_energy, blank_stare_metric, head_dynamics_stability, calibrated_vis_prob]
        Empirical fallback vector [0.48, 0.65, 0.95, 0.75] derived from medians of focused recordings.
        """
        if not self.face_detected or len(self.energy_buffer) < 30:
            return np.array([0.48, 0.65, 0.95, 0.75], dtype=np.float32)
            
        vis_prob = self.predict()
        if vis_prob == -1.0:
            return np.array([0.48, 0.65, 0.95, 0.75], dtype=np.float32)
            
        E = np.array(self.energy_buffer)
        
        # 1. Gaze energy from high-frequency singular variations
        gaze_energy = float(np.clip(np.std(E) / (np.mean(E) + 1e-6), 0.0, 1.0))
        
        # 2. Blank Stare Fixation Index (Bosch & D'Mello, 2021)
        blank_stare_metric = self._calculate_blank_stare_index()
        
        # 3. Head Pose Velocity & Jitter Stability (Bosch & D'Mello, 2021)
        head_dynamics_stability = self._calculate_head_dynamics()
        
        # 4. Calibrated visual engagement score
        calibrated_vis_prob = float(vis_prob)
        
        return np.array([gaze_energy, blank_stare_metric, head_dynamics_stability, calibrated_vis_prob], dtype=np.float32)

    def get_visual_quality(self):
        """
        Computes real-time visual signal quality Q_vis [0.0 - 1.0] (Wei et al., 2018).
        Factors in face presence, bounding box resolution, and tracking stability.
        
        Geometric Derivation of 160px:
        The ocular crop spans x in [0.1w, 0.9w] (width = 0.8w). At a face bounding-box
        width of w = 160px (at 640x480), the extracted crop width is exactly 0.8 * 160 = 128px,
        which matches the 128x96 grid width 1:1 horizontally (vertical is 1.5x upscaled from 64px to 96px).
        Scales dynamically with self.res_scale.
        """
        if not self.face_detected or len(self.head_pose_buffer) == 0:
            return 0.0
            
        latest_pose = self.head_pose_buffer[-1]
        face_w = latest_pose[2]
        
        # 1:1 horizontal sampling occurs at 160px * res_scale (crop width = 128px * res_scale)
        optimal_face_w = 160.0 * self.res_scale
        size_quality = float(np.clip(face_w / optimal_face_w, 0.4, 1.0))
        stability_quality = self._calculate_head_dynamics()
        
        # Linear combination: 60% spatial resolution adequacy, 40% motion stability
        # Theoretical range is [0.32, 1.0] when face is present (bounded to [0.0, 1.0])
        q_vis = float(np.clip(0.6 * size_quality + 0.4 * stability_quality, 0.0, 1.0))
        return q_vis

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
