import numpy as np
from collections import deque
import json
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Force CPU inference for stable real-time async execution
tf.config.set_visible_devices([], 'GPU')

class PhysioInferenceModel:
    def __init__(self, multires_path=r"../offline_training/multiresolution_cnn_lstm_model.h5",
                 multimodal_path=r"../offline_training/multimodal_lstm_model.h5",
                 physio_path=r"../offline_training/physio_lstm_model.h5"):
        self.hr_buffer = deque(maxlen=20)
        
        # Dual-branch rolling sequence buffers (10 timesteps window)
        self.physio_buffer = deque(maxlen=10)
        self.visual_buffer = deque(maxlen=10)
        self.sequence_buffer = deque(maxlen=10)
        
        # Personalized Calibration (3 minutes of readings)
        self.calibration_hr = deque(maxlen=180)
        self.calibration_rmssd = deque(maxlen=180)
        
        self.baseline_hr = 72.0
        self.baseline_rmssd = 30.0
        
        # Latest 4D visual image traits (data-derived empirical medians of focused recordings)
        self.latest_image_traits = np.array([0.48, 0.65, 0.95, 0.75], dtype=np.float32)
        
        # Pre-fill buffers with neutral zeros / baseline
        for _ in range(10):
            self.physio_buffer.append(np.zeros(3, dtype=np.float32))
            self.visual_buffer.append(np.array([0.48, 0.65, 0.95, 0.75], dtype=np.float32))
            self.sequence_buffer.append(np.zeros(7, dtype=np.float32))
            
        print("Loading Multimodal Multiresolution CNN-LSTM model (John et al., 2021)...")
        self.is_multires = False
        target_path = None
        
        if os.path.exists(multires_path):
            target_path = multires_path
            self.is_multires = True
        elif os.path.exists(multimodal_path):
            target_path = multimodal_path
        elif os.path.exists(physio_path):
            target_path = physio_path
            
        if target_path and os.path.exists(target_path):
            try:
                self.model = load_model(target_path)
                print(f"Model loaded successfully from {target_path} (Multiresolution Mode: {self.is_multires}).")
            except Exception as e:
                print(f"Error loading model: {e}. Fallback active.")
                self.model = None
        else:
            print("WARNING: Neural network model not found. Using baseline heuristic fallback.")
            self.model = None
            
        print("PhysioInferenceModel initialized (Multimodal Multiresolution CNN-LSTM with Watch Priority).")

        self.last_hr_timestamp = time.time()
        self.latest_rot_stability = 0.95

    def is_calibrating(self):
        return len(self.calibration_hr) < 30

    def update_image_traits(self, traits):
        """
        Updates the latest 4D image traits from Module A.
        In the absence of a dynamic Keras Masking layer, empirical median imputation
        [0.48, 0.65, 0.95, 0.75] is used as an interim workaround during camera occlusion
        to prevent artificial score collapse in the recurrent sequence buffer.
        """
        if traits is not None and len(traits) == 4:
            arr = np.array(traits, dtype=np.float32)
            if np.all(arr == 0):
                self.latest_image_traits = np.array([0.48, 0.65, 0.95, 0.75], dtype=np.float32)
            else:
                self.latest_image_traits = arr
        else:
            self.latest_image_traits = np.array([0.48, 0.65, 0.95, 0.75], dtype=np.float32)
            
        self.visual_buffer.append(self.latest_image_traits)

    def update_buffer(self, payload_str):
        """Processes watch payload and updates multiresolution temporal buffers."""
        try:
            data = json.loads(payload_str)
            if data.get("type") == "rotation_vector":
                x = abs(float(data.get("x", 0.0)))
                y = abs(float(data.get("y", 0.0)))
                z = abs(float(data.get("z", 0.0)))
                movement = x + y + z
                # High movement drops quality, calm wrist keeps quality ~0.95-1.0
                self.latest_rot_stability = float(np.clip(1.0 - (movement / 3.0), 0.35, 1.0))

            elif data.get("type") == "heart_rate":
                hr = data.get("value", 0.0)
                if hr > 0:
                    self.last_hr_timestamp = time.time()
                    self.hr_buffer.append(hr)
                    
                    rr_intervals = [60000.0 / h for h in self.hr_buffer]
                    sdnn = np.std(rr_intervals) if len(rr_intervals) > 1 else 0.0
                    
                    if len(rr_intervals) > 2:
                        diffs = np.diff(rr_intervals)
                        rmssd = np.sqrt(np.mean(diffs ** 2))
                    else:
                        rmssd = 0.0
                        
                    # Update calibration rolling baselines
                    self.calibration_hr.append(hr)
                    self.calibration_rmssd.append(rmssd)
                    self.baseline_hr = float(np.median(self.calibration_hr))
                    self.baseline_rmssd = float(np.median(self.calibration_rmssd))
                        
                    # Normalized Physio Features
                    delta_hr = (hr - self.baseline_hr) / 30.0
                    delta_rmssd = (rmssd - self.baseline_rmssd) / 50.0
                    sdnn_norm = min(1.0, sdnn / 20.0)
                    physio_vec = np.array([delta_hr, delta_rmssd, sdnn_norm], dtype=np.float32)
                    
                    self.physio_buffer.append(physio_vec)
                    
                    # 7D unified vector
                    multimodal_vec = np.concatenate([physio_vec, self.latest_image_traits]).astype(np.float32)
                    self.sequence_buffer.append(multimodal_vec)
        except Exception:
            pass

    def get_physio_quality(self):
        """
        Computes real-time physiological signal quality Q_phys [0.0 - 1.0] (Wei et al., 2018).
        High (1.0) when wrist is stable on desk; drops during violent hand movements to filter PPG noise.
        """
        if len(self.hr_buffer) == 0:
            return 0.0
            
        # If no recent reading in 5 seconds
        if time.time() - self.last_hr_timestamp > 5.0:
            return 0.20
            
        return float(np.clip(self.latest_rot_stability, 0.35, 1.0))

    def is_sensor_worn(self):
        """
        Determines whether the smartwatch is actively being worn on a wrist.
        Returns True if fresh, non-zero heart rate readings arrived in the last 5 seconds.
        """
        if len(self.hr_buffer) == 0:
            return False
        if time.time() - self.last_hr_timestamp > 5.0:
            return False
        return self.hr_buffer[-1] > 30.0

    def predict(self):
        """
        Computes focus probability [0.0 - 1.0] using Multiresolution CNN-LSTM.
        """
        if self.is_calibrating():
            return 0.50
            
        if self.model is None:
            return 0.70
            
        try:
            if self.is_multires:
                seq_p = np.array(self.physio_buffer, dtype=np.float32).reshape(1, 10, 3)
                seq_v = np.array(self.visual_buffer, dtype=np.float32).reshape(1, 10, 4)
                pred = self.model.predict({"input_physio": seq_p, "input_visual": seq_v}, verbose=0)
            else:
                seq = np.array(self.sequence_buffer, dtype=np.float32).reshape(1, 10, -1)
                pred = self.model.predict(seq, verbose=0)
                
            return float(np.clip(pred[0][0], 0.0, 1.0))
        except Exception as e:
            print(f"Inference error: {e}")
            return 0.65

    def get_latest_metrics(self):
        if len(self.hr_buffer) == 0:
            return 0.0, 0.0
        
        latest_hr = self.hr_buffer[-1]
        
        rr_intervals = [60000.0 / h for h in self.hr_buffer]
        if len(rr_intervals) > 2:
            diffs = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(diffs ** 2))
        else:
            rmssd = 0.0
            
        return float(latest_hr), float(rmssd)
