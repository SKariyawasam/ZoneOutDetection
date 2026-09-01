import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization, Concatenate, AveragePooling1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GroupKFold

# Force CPU execution
tf.config.set_visible_devices([], 'GPU')

sys.path.insert(0, os.path.dirname(__file__))
from train_multiresolution_cnn_lstm import build_multiresolution_dataset, SEQUENCE_LENGTH, NUM_PHYSIO_FEATURES, NUM_VISUAL_FEATURES

# Model 1: Physiological-Only Branch (Woo et al., 2018 CBAM Dual Pooling)
def build_physio_only_model():
    in_physio = Input(shape=(SEQUENCE_LENGTH, NUM_PHYSIO_FEATURES), name="input_physio")
    x = Conv1D(filters=16, kernel_size=2, padding="same", activation="relu")(in_physio)
    x = BatchNormalization()(x)
    p_avg = AveragePooling1D(pool_size=2, strides=1, padding="same")(x)
    p_max = MaxPooling1D(pool_size=2, strides=1, padding="same")(x)
    fused = Concatenate(axis=-1)([p_avg, p_max])
    fused = Dropout(0.2)(fused)
    lstm_out = LSTM(32, return_sequences=False)(fused)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    dense_1 = Dense(16, activation="relu")(lstm_out)
    out = Dense(1, activation="sigmoid", name="focus_output")(dense_1)
    m = Model(inputs=in_physio, outputs=out, name="Physio_Only_CNN_LSTM")
    m.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return m

# Model 2: Visual-Only Branch (Woo et al., 2018 CBAM Dual Pooling)
def build_visual_only_model():
    in_visual = Input(shape=(SEQUENCE_LENGTH, NUM_VISUAL_FEATURES), name="input_visual")
    x = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(in_visual)
    x = BatchNormalization()(x)
    v_avg = AveragePooling1D(pool_size=2, strides=1, padding="same")(x)
    v_max = MaxPooling1D(pool_size=2, strides=1, padding="same")(x)
    fused = Concatenate(axis=-1)([v_avg, v_max])
    fused = Dropout(0.2)(fused)
    lstm_out = LSTM(32, return_sequences=False)(fused)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    dense_1 = Dense(16, activation="relu")(lstm_out)
    out = Dense(1, activation="sigmoid", name="focus_output")(dense_1)
    m = Model(inputs=in_visual, outputs=out, name="Visual_Only_CNN_LSTM")
    m.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return m

# Model 3: Full Multimodal Branch (Woo et al., 2018 CBAM Dual Pooling)
def build_multimodal_model():
    in_physio = Input(shape=(SEQUENCE_LENGTH, NUM_PHYSIO_FEATURES), name="input_physio")
    x_p = Conv1D(filters=16, kernel_size=2, padding="same", activation="relu")(in_physio)
    x_p = BatchNormalization()(x_p)
    p_avg = AveragePooling1D(pool_size=2, strides=1, padding="same")(x_p)
    p_max = MaxPooling1D(pool_size=2, strides=1, padding="same")(x_p)
    x_p_fused = Concatenate(axis=-1)([p_avg, p_max])
    x_p_fused = Dropout(0.2)(x_p_fused)
    
    in_visual = Input(shape=(SEQUENCE_LENGTH, NUM_VISUAL_FEATURES), name="input_visual")
    x_v = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(in_visual)
    x_v = BatchNormalization()(x_v)
    v_avg = AveragePooling1D(pool_size=2, strides=1, padding="same")(x_v)
    v_max = MaxPooling1D(pool_size=2, strides=1, padding="same")(x_v)
    x_v_fused = Concatenate(axis=-1)([v_avg, v_max])
    x_v_fused = Dropout(0.2)(x_v_fused)
    
    fused = Concatenate(axis=-1)([x_p_fused, x_v_fused])
    lstm_out = LSTM(32, return_sequences=False)(fused)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    dense_1 = Dense(16, activation="relu")(lstm_out)
    out = Dense(1, activation="sigmoid", name="focus_output")(dense_1)
    m = Model(inputs=[in_physio, in_visual], outputs=out, name="Multimodal_CNN_LSTM")
    m.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return m

def evaluate_ablation():
    cache_path = os.path.join(os.path.dirname(__file__), "cached_multimodal_dataset.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, "rb") as f:
            X_p, X_v, y, groups = pickle.load(f)
    else:
        print("Extracting multimodal dataset...")
        X_p, X_v, y, groups = build_multiresolution_dataset()
        with open(cache_path, "wb") as f:
            pickle.dump((X_p, X_v, y, groups), f)
            
    print(f"\n=======================================================")
    print(f"Running Systematic Branch Ablation (Q-02) on {len(X_p)} samples across {len(np.unique(groups))} subjects")
    print(f"=======================================================\n")
    
    gkf = GroupKFold(n_splits=5)
    
    results = {
        "Physio-Only Branch": {"mse": [], "mae": [], "r2": []},
        "Visual-Only Branch": {"mse": [], "mae": [], "r2": []},
        "Full Multimodal Model": {"mse": [], "mae": [], "r2": []},
    }
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_p, y, groups=groups)):
        val_subjects = np.unique(groups[val_idx])
        print(f"\n--- [Fold {fold+1}/5] Test Subjects: {val_subjects} ---")
        y_val = y[val_idx]
        y_val_mean = np.mean(y_val)
        ss_tot = np.sum((y_val - y_val_mean) ** 2) + 1e-6
        
        # 1. Physio-Only
        m_p = build_physio_only_model()
        m_p.fit(X_p[train_idx], y[train_idx], epochs=12, batch_size=32, verbose=0)
        p_preds = m_p.predict(X_p[val_idx], verbose=0).flatten()
        p_mse = float(np.mean((y_val - p_preds) ** 2))
        p_mae = float(np.mean(np.abs(y_val - p_preds)))
        p_r2 = float(1.0 - (np.sum((y_val - p_preds) ** 2) / ss_tot))
        results["Physio-Only Branch"]["mse"].append(p_mse)
        results["Physio-Only Branch"]["mae"].append(p_mae)
        results["Physio-Only Branch"]["r2"].append(p_r2)
        print(f"  Physio-Only : MSE={p_mse:.6f}, MAE={p_mae:.4f}, R^2={p_r2:.4f}")
        
        # 2. Visual-Only
        m_v = build_visual_only_model()
        m_v.fit(X_v[train_idx], y[train_idx], epochs=12, batch_size=32, verbose=0)
        v_preds = m_v.predict(X_v[val_idx], verbose=0).flatten()
        v_mse = float(np.mean((y_val - v_preds) ** 2))
        v_mae = float(np.mean(np.abs(y_val - v_preds)))
        v_r2 = float(1.0 - (np.sum((y_val - v_preds) ** 2) / ss_tot))
        results["Visual-Only Branch"]["mse"].append(v_mse)
        results["Visual-Only Branch"]["mae"].append(v_mae)
        results["Visual-Only Branch"]["r2"].append(v_r2)
        print(f"  Visual-Only : MSE={v_mse:.6f}, MAE={v_mae:.4f}, R^2={v_r2:.4f}")
        
        # 3. Full Multimodal
        m_m = build_multimodal_model()
        m_m.fit({"input_physio": X_p[train_idx], "input_visual": X_v[train_idx]}, y[train_idx], epochs=12, batch_size=32, verbose=0)
        m_preds = m_m.predict({"input_physio": X_p[val_idx], "input_visual": X_v[val_idx]}, verbose=0).flatten()
        m_mse = float(np.mean((y_val - m_preds) ** 2))
        m_mae = float(np.mean(np.abs(y_val - m_preds)))
        m_r2 = float(1.0 - (np.sum((y_val - m_preds) ** 2) / ss_tot))
        results["Full Multimodal Model"]["mse"].append(m_mse)
        results["Full Multimodal Model"]["mae"].append(m_mae)
        results["Full Multimodal Model"]["r2"].append(m_r2)
        print(f"  Multimodal  : MSE={m_mse:.6f}, MAE={m_mae:.4f}, R^2={m_r2:.4f}")
        
    print("\n=======================================================")
    print("BRANCH ABLATION SUMMARY (5-Fold Subject-Wise Generalization):")
    print("=======================================================")
    for branch_name, m_dict in results.items():
        print(f"\n{branch_name}:")
        print(f"  Validation MSE : {np.mean(m_dict['mse']):.6f} ± {np.std(m_dict['mse']):.6f}")
        print(f"  Validation MAE : {np.mean(m_dict['mae']):.4f} ± {np.std(m_dict['mae']):.4f}")
        print(f"  Validation R^2 : {np.mean(m_dict['r2']):.4f} ± {np.std(m_dict['r2']):.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    evaluate_ablation()
