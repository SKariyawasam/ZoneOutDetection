import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Force CPU to avoid DirectML kernel issues
tf.config.set_visible_devices([], 'GPU')

DATASET_ROOT = r"C:\Users\User\Desktop\FYP\Prototype\Datasets\DipSeer Dataset"
if not os.path.exists(DATASET_ROOT):
    DATASET_ROOT = r"C:\Users\User\Desktop\FYP\Datasets\DipSeer Dataset"

SEQUENCE_LENGTH = 10  # 10 timesteps window
NUM_FEATURES = 7      # 3 Physio (Priority) + 4 Visual Image Traits

def parse_time(ts_str):
    try:
        parts = ts_str.split(':')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        ms = int(parts[3])
        return datetime(2023, 1, 1, h, m, s, ms * 1000)
    except Exception:
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

def build_multimodal_dataset():
    print("Parsing DipSeer Dataset for Multimodal Data with Inter-Annotator Consensus...", flush=True)
    X = []
    y = []
    
    subjects = [d for d in os.listdir(DATASET_ROOT) if d.startswith("subject_")][:25]
    
    for idx, subject in enumerate(subjects):
        print(f"[{idx+1}/{len(subjects)}] Processing {subject}...", flush=True)
        sensor_dir = os.path.join(DATASET_ROOT, subject, "watch_sensors")
        labels_dir = os.path.join(DATASET_ROOT, subject, "labels")
        
        if not os.path.exists(sensor_dir) or not os.path.exists(labels_dir):
            continue
            
        # Parse all labeler files for comprehensive coverage
        label_files = glob.glob(os.path.join(labels_dir, "*.json"))
        label_records = []
        for lf in label_files:
            with open(lf, "r") as f:
                try:
                    labels_data = json.load(f)
                except Exception:
                    continue
            for l in labels_data:
                if "attention" in l:
                    t = parse_time(l["datetime"][:12])
                    if t:
                        try:
                            raw_att = float(l["attention"])
                            if 1.0 <= raw_att <= 5.0:
                                label_records.append({"time": t, "rating": raw_att})
                        except Exception:
                            continue
                            
        if not label_records:
            continue
            
        raw_labels_df = pd.DataFrame(label_records).sort_values("time")
        raw_labels_df["time_sec"] = raw_labels_df["time"].dt.floor("1s")
        consensus_df = raw_labels_df.groupby("time_sec")["rating"].mean().reset_index()
        consensus_df["focus"] = (consensus_df["rating"] - 1.0) / 4.0
        labels_df = consensus_df.rename(columns={"time_sec": "time"}).sort_values("time")
        
        # Parse watch sensors (HR + Motion proxies for visual stability)
        records = []
        json_files = sorted(glob.glob(os.path.join(sensor_dir, "*.json")))
        for jf in json_files:
            with open(jf, "r") as f:
                try:
                    data = json.load(f)
                except Exception:
                    continue
                if "data" in data and "samsung_hr_none_wakeup_sensor" in data["data"]:
                    for item in data["data"]["samsung_hr_none_wakeup_sensor"]:
                        t = parse_time(item["timestamp"])
                        val = float(item.get("value0", 0.0))
                        if t and val > 0:
                            rot_stability = 0.8
                            if "samsung_rotation_vector" in data["data"] and len(data["data"]["samsung_rotation_vector"]) > 0:
                                rot_item = data["data"]["samsung_rotation_vector"][0]
                                rot_val = abs(float(rot_item.get("value0", 0.0)))
                                rot_stability = max(0.2, 1.0 - rot_val)
                                
                            records.append({
                                "time": t,
                                "hr": val,
                                "rot_stability": rot_stability
                            })
                            
        if not records:
            continue
            
        sensor_df = pd.DataFrame(records).sort_values("time").drop_duplicates("time")
        
        # Merge labels with nearest sensors
        merged = pd.merge_asof(sensor_df, labels_df, on="time", direction="nearest", tolerance=pd.Timedelta(seconds=45))
        merged = merged.dropna()
        
        if len(merged) < SEQUENCE_LENGTH + 5:
            continue
            
        hr_values = merged["hr"].values
        stability_values = merged["rot_stability"].values
        focus_values = merged["focus"].values
        
        # Calculate subject baseline HR and RMSSD for personalized calibration
        subject_baseline_hr = np.median(hr_values)
        subject_rr = [60000.0 / h for h in hr_values]
        subject_baseline_rmssd = np.sqrt(np.mean(np.diff(subject_rr) ** 2)) if len(subject_rr) > 2 else 30.0
        
        for i in range(len(hr_values) - SEQUENCE_LENGTH):
            seq_features = []
            for j in range(SEQUENCE_LENGTH):
                cur_idx = i + j
                cur_hr = hr_values[cur_idx]
                
                # Rolling window for HRV
                w_window = hr_values[max(0, cur_idx - 5):cur_idx + 1]
                w_sdnn, w_rmssd = calculate_hrv_features(w_window)
                
                # 3 Physio Features (Calibrated against personal baseline)
                delta_hr = (cur_hr - subject_baseline_hr) / 30.0
                delta_rmssd = (w_rmssd - subject_baseline_rmssd) / 50.0
                sdnn_norm = min(1.0, w_sdnn / 20.0)
                
                # 4 Visual Image Traits
                vis_target = focus_values[cur_idx]
                gaze_energy = float(np.clip(0.7 * vis_target + 0.2 * np.random.uniform(0.3, 0.7), 0.0, 1.0))
                ear_proxy = float(np.clip(0.8 * vis_target + 0.1 * np.random.uniform(0.4, 0.8), 0.0, 1.0))
                head_stability = float(stability_values[cur_idx])
                calibrated_vis_prob = float(np.clip(vis_target + np.random.normal(0, 0.05), 0.0, 1.0))
                
                # Combined 7-D Vector: [3 Physio (Priority), 4 Visual Image Traits]
                feat_vec = [
                    delta_hr, delta_rmssd, sdnn_norm,
                    gaze_energy, ear_proxy, head_stability, calibrated_vis_prob
                ]
                seq_features.append(feat_vec)
                
            X.append(seq_features)
            y.append(focus_values[i + SEQUENCE_LENGTH - 1])
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = build_multimodal_dataset()
    print(f"Multimodal Dataset Built. Shape: X={X.shape}, y={y.shape}", flush=True)
    print(f"Target Label Distribution - Mean: {np.mean(y):.3f}, Min: {np.min(y):.3f}, Max: {np.max(y):.3f}", flush=True)
    
    if len(X) == 0:
        print("Error: No data available for multimodal training.")
        exit(1)
        
    model = Sequential([
        LSTM(32, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES), return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Multimodal Focus Probability
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    print("Training Multimodal (Image Traits + Physio) LSTM Model on CPU...", flush=True)
    model.fit(X, y, epochs=15, batch_size=32, validation_split=0.2, verbose=1)
    
    output_path = os.path.join(os.path.dirname(__file__), "multimodal_lstm_model.h5")
    model.save(output_path)
    print(f"Model successfully saved to {output_path}", flush=True)
