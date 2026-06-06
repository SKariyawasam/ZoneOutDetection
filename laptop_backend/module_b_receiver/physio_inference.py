import numpy as np
from collections import deque
import json

class PhysioInferenceModel:
    def __init__(self, model_path=None):
        """
        Heuristic-based physiological focus scorer.
        Uses live HR and RMSSD (HRV) to compute a focus probability
        without requiring TensorFlow, avoiding DirectML deadlocks.
        
        Focused state signature:   moderate HR (65-80 bpm), low-medium RMSSD
        Zoned-out state signature: higher HR variability, elevated or dropping HR
        """
        self.hr_buffer = deque(maxlen=20)   # Rolling HR readings
        self.sequence_buffer = deque(maxlen=10)  # Rolling [hr, rmssd, sdnn] features
        
        # Pre-fill with neutral zeros
        for _ in range(10):
            self.sequence_buffer.append(np.zeros(3))
            
        print("PhysioInferenceModel initialized (heuristic mode - no TF required).")

    def update_buffer(self, payload_str):
        try:
            data = json.loads(payload_str)
            if data.get("type") == "heart_rate":
                hr = data.get("value", 0.0)
                if hr > 0:
                    self.hr_buffer.append(hr)
                    
                    # Calculate HRV features from rolling HR buffer
                    rr_intervals = [60000.0 / h for h in self.hr_buffer]
                    sdnn = np.std(rr_intervals) if len(rr_intervals) > 1 else 0.0
                    
                    if len(rr_intervals) > 2:
                        diffs = np.diff(rr_intervals)
                        rmssd = np.sqrt(np.mean(diffs ** 2))
                    else:
                        rmssd = 0.0
                        
                    self.sequence_buffer.append(np.array([hr, rmssd, sdnn]))
        except Exception:
            pass

    def predict(self):
        """
        Computes a focus probability [0.0 - 1.0] from recent HR/HRV data.
        
        Heuristic logic:
        - Focused: HR in calm range (60-85 bpm), low RMSSD (stable ANS)
        - Zoned out / stressed: high RMSSD or HR drifting out of normal range
        """
        if len(self.hr_buffer) < 3:
            return 0.5  # Not enough data yet

        recent_hrs = list(self.hr_buffer)[-10:]
        hr_mean = np.mean(recent_hrs)
        hr_std = np.std(recent_hrs)
        
        # Get latest rmssd
        latest = self.sequence_buffer[-1]
        rmssd = latest[1]
        
        # --- HR-based focus score ---
        # Focused HR zone: 60-85 bpm. Penalise deviation from centre (72 bpm).
        hr_center = 72.0
        hr_deviation = abs(hr_mean - hr_center)
        # Map 0 deviation → 1.0, 30+ deviation → 0.0
        hr_score = max(0.0, 1.0 - (hr_deviation / 30.0))
        
        # --- Variability penalty ---
        # High HR std within short window = mental wandering / drowsiness
        # 0 std → 1.0, 10+ std → 0.0
        stability_score = max(0.0, 1.0 - (hr_std / 10.0))
        
        # --- RMSSD score ---
        # Moderate RMSSD (20-50ms) is healthy focus. Very high (>80ms) = zoned out.
        if rmssd < 1.0:
            rmssd_score = 0.5  # No data yet, neutral
        elif rmssd <= 50.0:
            rmssd_score = 1.0 - (rmssd / 100.0)  # Lower RMSSD = more focused
        else:
            rmssd_score = max(0.0, 1.0 - (rmssd / 80.0))
            
        # Weighted combination
        focus_prob = (0.5 * hr_score) + (0.3 * stability_score) + (0.2 * rmssd_score)
        return float(np.clip(focus_prob, 0.0, 1.0))

    def get_latest_metrics(self):
        """Returns (hr, rmssd) from the most recent physiological reading."""
        if len(self.sequence_buffer) == 0 or np.all(self.sequence_buffer[-1] == 0):
            return 0.0, 0.0
        latest = self.sequence_buffer[-1]
        return float(latest[0]), float(latest[1])

