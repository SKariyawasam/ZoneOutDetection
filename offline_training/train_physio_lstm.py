import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# To avoid CudnnRNN missing kernel crashes with DirectML on Windows,
# and because the dataset is tiny (2856 samples), we train on CPU.
tf.config.set_visible_devices([], 'GPU')

DATASET_ROOT = r"C:\Users\User\Desktop\FYP\Datasets\DipSeer Dataset"
SEQUENCE_LENGTH = 10 # 10 timesteps of HR data

def parse_time(ts_str):
    # Format: HH:MM:SS:fff
    try:
        parts = ts_str.split(':')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        ms = int(parts[3])
        # use a dummy date
        return datetime(2023, 1, 1, h, m, s, ms * 1000)
    except:
        return None

def calculate_hrv_features(hr_buffer):
    rr_intervals = [60000.0 / h if h > 0 else 0 for h in hr_buffer]
    sdnn = np.std(rr_intervals) if len(rr_intervals) > 1 else 0.0
    if len(rr_intervals) > 2:
        diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(diffs ** 2))
    else:
        rmssd = 0.0
    return sdnn, rmssd

def build_dataset():
    print("Parsing DipSeer Dataset for physiological data...")
    X = []
    y = []
    
    subjects = [d for d in os.listdir(DATASET_ROOT) if d.startswith("subject_")]
    
    for subject in subjects:
        sensor_dir = os.path.join(DATASET_ROOT, subject, "watch_sensors")
        labels_file = os.path.join(DATASET_ROOT, subject, "labels", "self_labeling.json")
        
        if not os.path.exists(sensor_dir) or not os.path.exists(labels_file):
            continue
            
        with open(labels_file, "r") as f:
            try:
                labels_data = json.load(f)
            except:
                continue
                
        # Parse labels into a time-series dataframe
        label_records = []
        for l in labels_data:
            if "attention" in l:
                t = parse_time(l["datetime"][:12]) # trim microseconds to ms length roughly
                if t:
                    # Map attention 1-5 to probability 0.0 - 1.0 (1=0.0, 3=0.5, 5=1.0)
                    att = float(l["attention"])
                    prob = (att - 1.0) / 4.0
                    label_records.append({"time": t, "focus": prob})
                    
        if not label_records:
            continue
            
        labels_df = pd.DataFrame(label_records).sort_values("time")
        
        # Parse watch sensors
        hr_records = []
        json_files = sorted(glob.glob(os.path.join(sensor_dir, "*.json")))
        for jf in json_files:
            with open(jf, "r") as f:
                try:
                    data = json.load(f)
                except:
                    continue
                if "data" in data and "samsung_hr_none_wakeup_sensor" in data["data"]:
                    for item in data["data"]["samsung_hr_none_wakeup_sensor"]:
                        t = parse_time(item["timestamp"])
                        val = float(item["value0"])
                        if t and val > 0:
                            hr_records.append({"time": t, "hr": val})
                            
        if not hr_records:
            continue
            
        hr_df = pd.DataFrame(hr_records).sort_values("time").drop_duplicates("time")
        
        # Merge labels with nearest HR (forward fill backwards)
        merged = pd.merge_asof(hr_df, labels_df, on="time", direction="nearest", tolerance=pd.Timedelta(seconds=30))
        merged = merged.dropna()
        
        # Build sequences
        hr_values = merged["hr"].values
        focus_values = merged["focus"].values
        
        for i in range(len(hr_values) - SEQUENCE_LENGTH):
            seq_hr = hr_values[i:i+SEQUENCE_LENGTH]
            sdnn, rmssd = calculate_hrv_features(seq_hr)
            
            # Feature vector at each timestep could be [HR, RMSSD, SDNN], but we'll use HR for the sequence, and global HRV
            # Actually, let's make sequence of [HR_t, RMSSD_t] by calculating rolling HRV
            
            seq_features = []
            for j in range(SEQUENCE_LENGTH):
                window = hr_values[max(0, i+j-5):i+j+1] # small rolling window
                w_sdnn, w_rmssd = calculate_hrv_features(window)
                seq_features.append([seq_hr[j], w_rmssd, w_sdnn])
                
            X.append(seq_features)
            y.append(focus_values[i+SEQUENCE_LENGTH-1])
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = build_dataset()
    print(f"Dataset built. Shape: X={X.shape}, y={y.shape}")
    
    if len(X) == 0:
        print("No valid data found to train LSTM.")
        exit(1)
        
    model = Sequential([
        LSTM(32, input_shape=(SEQUENCE_LENGTH, 3), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid') # Focus probability
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    print("Training LSTM on GPU...")
    model.fit(X, y, epochs=15, batch_size=32, validation_split=0.2)
    
    model.save("physio_lstm_model.h5")
    print("Model saved to physio_lstm_model.h5")
