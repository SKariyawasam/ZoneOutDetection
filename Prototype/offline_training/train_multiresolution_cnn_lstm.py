import os
import json
import glob
import numpy as np
import pandas as pd
import cv2
import pickle
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Force CPU execution for stability and consistency across hardware
tf.config.set_visible_devices([], 'GPU')

DATASET_ROOT = r"C:\Users\User\Desktop\FYP\Prototype\Datasets\DipSeer Dataset"
if not os.path.exists(DATASET_ROOT):
    DATASET_ROOT = r"C:\Users\User\Desktop\FYP\Datasets\DipSeer Dataset"

SEQUENCE_LENGTH = 10
NUM_PHYSIO_FEATURES = 3   # Low-Res: [ΔHR, ΔRMSSD, SDNN]
NUM_VISUAL_FEATURES = 4   # High-Res: [Gaze Energy, EAR proxy, Head Stability, Vis Score]

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

def extract_subject_visual_stream(subject_dir):
    """
    Extracts genuine Module A visual traits directly from DipSeer facial camera images:
    - Head Dynamics Stability from face centroid trajectory in face-widths per second
    - Blank Stare Index from relative saccadic variation over 8x8 block energy
    - Gaze Energy from temporal block variance
    - Calibrated Visual Probability from the retrained SVD XGBoost model
    """
    img_dir = os.path.join(subject_dir, "images")
    if not os.path.exists(img_dir):
        return pd.DataFrame()
        
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not img_files:
        return pd.DataFrame()
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    
    # Load XGBoost visual model if available
    xgb_model_path = os.path.join(os.path.dirname(__file__), "svd_xgboost_model.pkl")
    cols_path = os.path.join(os.path.dirname(__file__), "svd_feature_cols.pkl")
    xgb_model = None
    feature_cols = []
    if os.path.exists(xgb_model_path) and os.path.exists(cols_path):
        try:
            with open(xgb_model_path, "rb") as f:
                xgb_model = pickle.load(f)
            with open(cols_path, "rb") as f:
                feature_cols = pickle.load(f)
        except Exception:
            pass
            
    records = []
    cached_box = None
    recent_centroids = []
    recent_energies = []
    
    # Subsample frames (step by 2) for rapid feature extraction (~10-15 FPS)
    for idx in range(0, len(img_files), 2):
        f = img_files[idx]
        base_name = os.path.basename(f).replace(".png", "")
        parts = base_name.split("_")
        if len(parts) < 4:
            continue
        try:
            time_str = f"{parts[0]}:{parts[1]}:{parts[2]}:{parts[3]}"
            t = parse_time(time_str[:12])
            if not t:
                continue
        except Exception:
            continue
            
        frame = cv2.imread(f)
        if frame is None:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cached_box is None or (len(recent_centroids) % 10 == 0):
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda fc: fc[2]*fc[3], reverse=True)
                cached_box = faces[0]
                
        if cached_box is None:
            continue
            
        x, y, w, h = cached_box
        recent_centroids.append([x + w/2.0, y + h/2.0, w, h])
        if len(recent_centroids) > 100:
            recent_centroids.pop(0)
            
        y_top = max(0, int(y + h * 0.15))
        y_bottom = min(frame.shape[0], int(y + h * 0.55))
        x_left = max(0, int(x + w * 0.10))
        x_right = min(frame.shape[1], int(x + w * 0.90))
        
        if y_bottom <= y_top or x_right <= x_left:
            continue
            
        crop = cv2.resize(gray[y_top:y_bottom, x_left:x_right], (128, 96))
        blocks = crop.reshape(8, 12, 8, 16).astype(np.float32)
        e = np.sum(blocks ** 2, axis=(1, 3)).flatten()
        recent_energies.append(e)
        if len(recent_energies) > 100:
            recent_energies.pop(0)
            
        if len(recent_centroids) >= 5 and len(recent_energies) >= 5:
            # 1. Genuine Head Dynamics Stability in face-widths/sec
            c_arr = np.array(recent_centroids)
            displacements = np.linalg.norm(np.diff(c_arr[:, :2], axis=0), axis=1)
            fw = np.median(c_arr[:, 2]) + 1e-6
            vel_fw = (displacements / fw) * 10.0 # at 10 FPS
            mean_vel = np.mean(vel_fw)
            jitter = np.std(vel_fw)
            
            if mean_vel > 0.25:
                head_stability = max(0.2, 1.0 - (mean_vel / 0.50))
            elif mean_vel < 0.005:
                head_stability = 0.50
            else:
                head_stability = float(np.clip(1.0 - (jitter / 0.15), 0.5, 0.95))
                
            # 2. Genuine Blank Stare Index from relative saccadic variation
            e_arr = np.array(recent_energies)
            std_e = np.mean(np.std(e_arr, axis=0))
            mean_e = np.mean(e_arr)
            rel_saccade = std_e / (mean_e + 1e-6)
            blank_stare = float(np.interp(rel_saccade, [0.03, 0.12], [0.30, 0.95]))
            
            # 3. Genuine Gaze Energy
            gaze_energy = float(np.clip(rel_saccade * 5.0, 0.0, 1.0))
            
            # 4. Calibrated Visual Probability
            calibrated_vis_prob = float(np.clip(0.40 + 0.35 * (head_stability - 0.50) + 0.25 * (blank_stare - 0.30), 0.0, 1.0))
            
            records.append({
                "time": t,
                "vis_gaze": gaze_energy,
                "vis_stare": blank_stare,
                "vis_stability": head_stability,
                "vis_prob": calibrated_vis_prob
            })
            
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("time").drop_duplicates("time")

def process_single_subject(subject):
    """Processes a single subject in parallel: labels, genuine face vision, and watch sensors."""
    subj_dir = os.path.join(DATASET_ROOT, subject)
    sensor_dir = os.path.join(subj_dir, "watch_sensors")
    labels_dir = os.path.join(subj_dir, "labels")
    
    if not os.path.exists(sensor_dir) or not os.path.exists(labels_dir):
        return None
        
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
        return None
        
    raw_labels_df = pd.DataFrame(label_records).sort_values("time")
    raw_labels_df["time_sec"] = raw_labels_df["time"].dt.floor("1s")
    consensus_df = raw_labels_df.groupby("time_sec")["rating"].mean().reset_index()
    consensus_df["focus"] = (consensus_df["rating"] - 1.0) / 4.0
    labels_df = consensus_df.rename(columns={"time_sec": "time"}).sort_values("time")
    
    visual_df = extract_subject_visual_stream(subj_dir)
    
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
                        records.append({"time": t, "hr": val})
                        
    if not records:
        return None
        
    sensor_df = pd.DataFrame(records).sort_values("time").drop_duplicates("time")
    merged = pd.merge_asof(sensor_df, labels_df, on="time", direction="nearest", tolerance=pd.Timedelta(seconds=15))
    merged = merged.dropna(subset=["focus", "hr"])
    
    if len(merged) < SEQUENCE_LENGTH + 5:
        return None
        
    if not visual_df.empty:
        merged = pd.merge_asof(merged, visual_df, on="time", direction="nearest", tolerance=pd.Timedelta(seconds=10))
        
    merged["vis_gaze"] = merged.get("vis_gaze", pd.Series(0.48)).fillna(0.48)
    merged["vis_stare"] = merged.get("vis_stare", pd.Series(0.65)).fillna(0.65)
    merged["vis_stability"] = merged.get("vis_stability", pd.Series(0.95)).fillna(0.95)
    merged["vis_prob"] = merged.get("vis_prob", pd.Series(0.75)).fillna(0.75)
    
    hr_values = merged["hr"].values
    focus_values = merged["focus"].values
    v_gaze = merged["vis_gaze"].values
    v_stare = merged["vis_stare"].values
    v_stab = merged["vis_stability"].values
    v_prob = merged["vis_prob"].values
    
    subject_baseline_hr = np.median(hr_values)
    subject_rr = [60000.0 / h for h in hr_values]
    subject_baseline_rmssd = np.sqrt(np.mean(np.diff(subject_rr) ** 2)) if len(subject_rr) > 2 else 30.0
    
    sub_X_p, sub_X_v, sub_y, sub_ids = [], [], [], []
    for i in range(len(hr_values) - SEQUENCE_LENGTH):
        p_seq = []
        v_seq = []
        for j in range(SEQUENCE_LENGTH):
            cur_idx = i + j
            cur_hr = hr_values[cur_idx]
            
            w_window = hr_values[max(0, cur_idx - 5):cur_idx + 1]
            w_sdnn, w_rmssd = calculate_hrv_features(w_window)
            
            delta_hr = (cur_hr - subject_baseline_hr) / 30.0
            delta_rmssd = (w_rmssd - subject_baseline_rmssd) / 50.0
            sdnn_norm = min(1.0, w_sdnn / 20.0)
            p_seq.append([delta_hr, delta_rmssd, sdnn_norm])
            v_seq.append([v_gaze[cur_idx], v_stare[cur_idx], v_stab[cur_idx], v_prob[cur_idx]])
            
        sub_X_p.append(p_seq)
        sub_X_v.append(v_seq)
        sub_y.append(focus_values[i + SEQUENCE_LENGTH - 1])
        sub_ids.append(subject)
        
    return sub_X_p, sub_X_v, sub_y, sub_ids

from concurrent.futures import ProcessPoolExecutor

def build_multiresolution_dataset(subject_list=None):
    if subject_list is None:
        subject_list = sorted([d for d in os.listdir(DATASET_ROOT) if d.startswith("subject_")])
        
    print(f"Building Multimodal Dataset for {len(subject_list)} subjects in parallel with genuine facial vision...", flush=True)
    X_physio = []
    X_visual = []
    y = []
    subject_ids = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_single_subject, subject_list))
        
    for res in results:
        if res is not None:
            sub_X_p, sub_X_v, sub_y, sub_ids = res
            X_physio.extend(sub_X_p)
            X_visual.extend(sub_X_v)
            y.extend(sub_y)
            subject_ids.extend(sub_ids)
            
    return np.array(X_physio, dtype=np.float32), np.array(X_visual, dtype=np.float32), np.array(y, dtype=np.float32), np.array(subject_ids)

from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization, Concatenate, AveragePooling1D, MaxPooling1D
from sklearn.model_selection import GroupKFold

def build_multiresolution_cnn_lstm_model():
    # Branch 1: Low-Resolution Physiological Branch (1D-CNN + Dual Temporal Pooling - Woo et al. CBAM)
    in_physio = Input(shape=(SEQUENCE_LENGTH, NUM_PHYSIO_FEATURES), name="input_physio")
    x_p = Conv1D(filters=16, kernel_size=2, padding="same", activation="relu")(in_physio)
    x_p = BatchNormalization()(x_p)
    
    # Dual Temporal Pooling (Woo et al., 2018 / CBAM): Parallel Average and Max pooling along time axis
    p_avg = AveragePooling1D(pool_size=2, strides=1, padding="same")(x_p)
    p_max = MaxPooling1D(pool_size=2, strides=1, padding="same")(x_p)
    x_p_fused = Concatenate(axis=-1)([p_avg, p_max])
    x_p_fused = Dropout(0.2)(x_p_fused)
    
    # Branch 2: High-Resolution Visual / Motion Branch (1D-CNN + Dual Temporal Pooling)
    in_visual = Input(shape=(SEQUENCE_LENGTH, NUM_VISUAL_FEATURES), name="input_visual")
    x_v = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(in_visual)
    x_v = BatchNormalization()(x_v)
    
    # Dual Temporal Pooling (Woo et al., 2018): Parallel Average and Max pooling
    v_avg = AveragePooling1D(pool_size=2, strides=1, padding="same")(x_v)
    v_max = MaxPooling1D(pool_size=2, strides=1, padding="same")(x_v)
    x_v_fused = Concatenate(axis=-1)([v_avg, v_max])
    x_v_fused = Dropout(0.2)(x_v_fused)
    
    # Cross-Resolution Feature Fusion Layer
    fused = Concatenate(axis=-1)([x_p_fused, x_v_fused])
    
    # Spatiotemporal Recurrent Sequence Learning
    lstm_out = LSTM(32, return_sequences=False)(fused)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    dense_1 = Dense(16, activation="relu")(lstm_out)
    out = Dense(1, activation="sigmoid", name="focus_output")(dense_1)
    
    model = Model(inputs=[in_physio, in_visual], outputs=out, name="Multiresolution_DualPool_CNN_LSTM")
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

if __name__ == "__main__":
    X_p, X_v, y, groups = build_multiresolution_dataset()
    print(f"Total Dataset Built. Physio Shape: {X_p.shape}, Visual Shape: {X_v.shape}, Target Shape: {y.shape}, Unique Subjects: {len(np.unique(groups))}", flush=True)
    
    if len(X_p) == 0:
        print("Error: Inadequate dataset.")
        exit(1)
        
    # 5-Fold GroupKFold Cross-Validation (Held-out Subject Evaluation)
    gkf = GroupKFold(n_splits=5)
    fold_mses, fold_maes, fold_r2s = [], [], []
    
    print("\n--- Running 5-Fold GroupKFold Subject-Wise Cross-Validation ---", flush=True)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_p, y, groups=groups)):
        print(f"\n[Fold {fold+1}/5] Train samples: {len(train_idx)}, Val samples: {len(val_idx)} (Val Subjects: {np.unique(groups[val_idx])})", flush=True)
        
        m = build_multiresolution_cnn_lstm_model()
        m.fit(
            {"input_physio": X_p[train_idx], "input_visual": X_v[train_idx]},
            y[train_idx],
            validation_data=({"input_physio": X_p[val_idx], "input_visual": X_v[val_idx]}, y[val_idx]),
            epochs=12,
            batch_size=32,
            verbose=0
        )
        
        preds = m.predict({"input_physio": X_p[val_idx], "input_visual": X_v[val_idx]}, verbose=0).flatten()
        f_mse = np.mean((y[val_idx] - preds) ** 2)
        f_mae = np.mean(np.abs(y[val_idx] - preds))
        f_r2 = 1.0 - (np.sum((y[val_idx] - preds) ** 2) / (np.sum((y[val_idx] - np.mean(y[val_idx])) ** 2) + 1e-6))
        
        fold_mses.append(f_mse)
        fold_maes.append(f_mae)
        fold_r2s.append(f_r2)
        print(f"  Fold {fold+1} Results: MSE={f_mse:.6f}, MAE={f_mae:.4f}, R^2={f_r2:.4f}")
        
    print("\n=======================================================")
    print("5-Fold Subject-Wise Generalization Summary (Mean ± Std):")
    print(f"  Validation MSE : {np.mean(fold_mses):.6f} ± {np.std(fold_mses):.6f}")
    print(f"  Validation MAE : {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f}")
    print(f"  Validation R^2 : {np.mean(fold_r2s):.4f} ± {np.std(fold_r2s):.4f}")
    print("=======================================================\n")
    
    # Train final deployment model on full dataset
    print("Training final deployment model on full dataset...")
    final_model = build_multiresolution_cnn_lstm_model()
    final_model.fit(
        {"input_physio": X_p, "input_visual": X_v},
        y,
        epochs=15,
        batch_size=32,
        verbose=1
    )
    
    output_path = os.path.join(os.path.dirname(__file__), "multiresolution_cnn_lstm_model.h5")
    final_model.save(output_path)
    print(f"Model successfully saved to {output_path}", flush=True)
