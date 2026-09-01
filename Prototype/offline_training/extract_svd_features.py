import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tsfel
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Setup TSFEL config
cfg = tsfel.get_features_by_domain()
if 'temporal' in cfg:
    del cfg['temporal'] # Only statistical and spectral as per paper

def extract_eye_energy_matrix(video_path):
    """
    Reads a video and extracts the SVD energy matrix E matching live inference:
    - Upper-face geometric band y in [0.15h, 0.55h], x in [0.1w, 0.9w]
    - Exactly 200 frames (10 seconds at 20 FPS)
    - Vectorized 8x8 block energy computation with periodic face tracking
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    frames_energy = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    
    cached_box = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if frame_idx % 3 == 0:
            continue # Subsample to 20 FPS
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face initially, and refresh every 15 frames for tracking stability
        if cached_box is None or (len(frames_energy) % 15 == 0):
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                cached_box = faces[0]
                
        if cached_box is None:
            frames_energy.append(np.zeros(64, dtype=np.float32))
            continue
            
        x, y, w, h = cached_box
        y_top = max(0, int(y + h * 0.15))
        y_bottom = min(frame.shape[0], int(y + h * 0.55))
        x_left = max(0, int(x + w * 0.10))
        x_right = min(frame.shape[1], int(x + w * 0.90))
        
        if y_bottom <= y_top or x_right <= x_left:
            frames_energy.append(np.zeros(64, dtype=np.float32))
            continue
            
        eye_crop = gray[y_top:y_bottom, x_left:x_right]
        if eye_crop.size == 0:
            frames_energy.append(np.zeros(64, dtype=np.float32))
            continue
            
        eye_crop = cv2.resize(eye_crop, (128, 96)) # Width 128, Height 96
        
        # Vectorized 8x8 block energy aggregation (each block 12x16 pixels)
        blocks = eye_crop.reshape(8, 12, 8, 16).astype(np.float32)
        energies = np.sum(blocks ** 2, axis=(1, 3)).flatten()
        frames_energy.append(energies)
        
        if len(frames_energy) >= 200:
            break
            
    cap.release()
    
    # Enforce exactly 200 frames (matching live 200-row SVD rolling buffer)
    if len(frames_energy) < 200:
        return None
        
    return np.array(frames_energy[:200], dtype=np.float32) # Shape: (200, 64)

def process_video(args):
    video_path, label = args
    try:
        E = extract_eye_energy_matrix(video_path)
        if E is None:
            return None
            
        # CPU accelerated NumPy SVD (matching live SVD CPU execution)
        U, S, Vt = np.linalg.svd(E, full_matrices=False)
        U_np = U
        
        best_vector = None
        max_amplitude = -1
        
        # Pick the vector with highest high-frequency amplitude
        for i in range(1, U_np.shape[1]):
            u_i = U_np[:, i]
            # scale to [0, 1]
            u_i = (u_i - np.min(u_i)) / (np.max(u_i) - np.min(u_i) + 1e-6)
            fft_vals = np.abs(np.fft.rfft(u_i))
            hf_amp = np.max(fft_vals[len(fft_vals)//2:])
            if hf_amp > max_amplitude:
                max_amplitude = hf_amp
                best_vector = u_i
                
        if best_vector is None:
            best_vector = U_np[:, 0]
            
        # TSFEL extraction at fs=20 (matching live inference 20 FPS)
        features = tsfel.time_series_features_extractor(cfg, best_vector, fs=20, verbose=0)
        
        feature_dict = features.iloc[0].to_dict()
        feature_dict['label'] = int(label)
        feature_dict['video_path'] = video_path
        return feature_dict
    except Exception as e:
        return None

if __name__ == "__main__":
    print("Starting SVD Feature Extraction (200-frame fixed window, live-matched upper-face region)...", flush=True)
    labels_path = r"c:\Users\User\Desktop\FYP\Prototype\Datasets\DAiSEE Dataset\Labels\TrainLabels.csv"
    if not os.path.exists(labels_path):
        labels_path = r"c:\Users\User\Desktop\FYP\Datasets\DAiSEE Dataset\Labels\TrainLabels.csv"
        
    labels_df = pd.read_csv(labels_path)
    labels_df['FocusLabel'] = labels_df['Engagement'].apply(lambda x: 1 if x >= 2 else 0)
    
    base_dir = r"c:\Users\User\Desktop\FYP\Prototype\Datasets\DAiSEE Dataset\DataSet\Train"
    if not os.path.exists(base_dir):
        base_dir = r"c:\Users\User\Desktop\FYP\Datasets\DAiSEE Dataset\DataSet\Train"
        
    # Gather all minority (FocusLabel=0) and majority (FocusLabel=1) videos on disk
    minority_args = []
    majority_args = []
    
    for idx, row in labels_df.iterrows():
        clip_id = str(row['ClipID']).replace('.avi', '')
        subject_id = clip_id[:6]
        vid_path = os.path.join(base_dir, subject_id, clip_id, f"{clip_id}.avi")
        if os.path.exists(vid_path):
            if row['FocusLabel'] == 0:
                minority_args.append((vid_path, 0))
            else:
                majority_args.append((vid_path, 1))
                
    # Balance: Take all available minority clips (234) + 3x majority clips (~700 clips) for a robust 1:3 ratio
    import random
    random.seed(42)
    sample_majority = random.sample(majority_args, min(len(majority_args), len(minority_args) * 3))
    video_args = minority_args + sample_majority
    random.shuffle(video_args)
    
    print(f"Total videos queued for extraction: {len(video_args)} (Minority Zoned-out: {len(minority_args)}, Majority Focused: {len(sample_majority)})", flush=True)
    
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in tqdm(executor.map(process_video, video_args), total=len(video_args)):
            if res is not None:
                results.append(res)
                
    out_df = pd.DataFrame(results)
    out_path = os.path.join(os.path.dirname(__file__), "svd_features_train.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nExtraction complete. Saved {len(out_df)} samples to {out_path}.", flush=True)
    print("Extracted class counts:\n", out_df['label'].value_counts(), flush=True)
