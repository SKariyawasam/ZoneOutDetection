import numpy as np
import tensorflow as tf
from collections import deque
import json
import os

class PhysioInferenceModel:
    def __init__(self, model_path="c:/Users/User/Desktop/FYP/DipSEER Code/models/physio_lstm_model.h5"):
        """Loads the LSTM network pre-trained on DipSEER."""
        self.time_steps = 10
        self.feature_dim = 3
        
        self.hr_buffer = deque(maxlen=20) # to calculate rolling rmssd/sdnn
        self.sequence_buffer = deque(maxlen=self.time_steps)
        
        # Pre-fill sequence buffer with zeros
        for _ in range(self.time_steps):
            self.sequence_buffer.append(np.zeros(self.feature_dim))
            
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"Successfully loaded physio model from {model_path}")
            else:
                self.model = None
                print(f"Warning: Physio model not found at {model_path}. Using fallback mock inferences.")
        except Exception as e:
            print(f"Error loading physio model: {e}")
            self.model = None

    def update_buffer(self, payload_str):
        try:
            data = json.loads(payload_str)
            if data.get("type") == "heart_rate":
                hr = data.get("value", 70.0)
                if hr > 0:
                    self.hr_buffer.append(hr)
                    
                    # Calculate HRV features
                    rr_intervals = [60000.0 / h for h in self.hr_buffer]
                    sdnn = np.std(rr_intervals) if len(rr_intervals) > 1 else 0.0
                    
                    if len(rr_intervals) > 2:
                        diffs = np.diff(rr_intervals)
                        rmssd = np.sqrt(np.mean(diffs ** 2))
                    else:
                        rmssd = 0.0
                        
                    # Append to sequence buffer
                    self.sequence_buffer.append(np.array([hr, rmssd, sdnn]))
        except Exception as e:
            pass

    def predict(self):
        """
        Processes temporal physiological signals.
        Returns: float (0.0 to 1.0) predicting cognitive/affective state.
        """
        if self.model is None:
            # Mock fallback if model couldn't be loaded
            return 0.60
            
        if len(self.sequence_buffer) < self.time_steps:
            return 0.5 # Not enough data
            
        # Shape: (1, time_steps, feature_dim)
        input_data = np.array([self.sequence_buffer]) 
        
        try:
            prediction = self.model.predict(input_data, verbose=0)
            return float(prediction[0][0])
        except Exception as e:
            print(f"Error during physio inference: {e}")
            return 0.5
