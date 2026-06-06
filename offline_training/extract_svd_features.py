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
    """Reads a video and extracts the SVD energy matrix E"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    frames_energy = []
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 3 == 0:
            continue # Skip 1 in every 3 frames (20 FPS)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(eyes) == 0:
            frames_energy.append(np.zeros(64)) # Fallback if no eye found
            continue
            
        # Use the largest detected eye
        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)
        x, y, w, h = eyes[0]
        
        # 25% padding
        pad_x = w * 0.25
        pad_y = h * 0.25
        
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(frame.shape[1], int(x + w + pad_x))
        y2 = min(frame.shape[0], int(y + h + pad_y))
        
        if x2 <= x1 or y2 <= y1:
            frames_energy.append(np.zeros(64))
            continue
            
        eye_crop = gray[y1:y2, x1:x2]
        eye_crop = cv2.resize(eye_crop, (128, 96)) # Width x Height (96x128 in paper is likely HeightxWidth)
        
        # Divide into 8x8 blocks (each block is 12x16)
        block_energies = []
        for row in range(8):
            for col in range(8):
                block = eye_crop[row*12:(row+1)*12, col*16:(col+1)*16]
                energy = np.sum(block.astype(np.float32) ** 2)
                block_energies.append(energy)
                
        frames_energy.append(np.array(block_energies))
            
    cap.release()
    if len(frames_energy) == 0:
        return None
        
    return np.array(frames_energy) # Shape: (k, 64)

def process_video(args):
    video_path, label = args
    try:
        E = extract_eye_energy_matrix(video_path)
        if E is None or len(E) < 10:
            return None
            
        # GPU accelerated SVD
        E_tensor = tf.convert_to_tensor(E, dtype=tf.float32)
        S, U, V = tf.linalg.svd(E_tensor)
        U_np = U.numpy()
        
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
            
        # TSFEL extraction
        features = tsfel.time_series_features_extractor(cfg, best_vector, fs=30, verbose=0)
        
        feature_dict = features.iloc[0].to_dict()
        feature_dict['label'] = label
        feature_dict['video_path'] = video_path
        return feature_dict
    except Exception as e:
        print(f"Failed processing {video_path}: {e}")
        return None

if __name__ == "__main__":
    print("Starting SVD Feature Extraction...")
    labels_df = pd.read_csv(r"c:\Users\User\Desktop\FYP\Datasets\DAiSEE Dataset\Labels\TrainLabels.csv")
    
    # Map engagement to binary Focus label:
    # 0, 1 = Zoned Out (0)
    # 2, 3 = Focused (1)
    labels_df['FocusLabel'] = labels_df['Engagement'].apply(lambda x: 1 if x >= 2 else 0)
    
    # For proof of concept / speed, we sample a subset. Change frac=1.0 for full dataset.
    sample_df = labels_df.sample(frac=0.1, random_state=42)
    
    video_args = []
    base_dir = r"c:\Users\User\Desktop\FYP\Datasets\DAiSEE Dataset\DataSet\Train"
    
    for idx, row in sample_df.iterrows():
        clip_id = str(row['ClipID']).replace('.avi', '')
        # Subject ID is first 6 characters
        subject_id = clip_id[:6]
        vid_path = os.path.join(base_dir, subject_id, clip_id, f"{clip_id}.avi")
        if os.path.exists(vid_path):
            video_args.append((vid_path, row['FocusLabel']))
            
    print(f"Found {len(video_args)} videos to process.")
    
    results = []
    # Multiprocessing across CPU cores
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in tqdm(executor.map(process_video, video_args), total=len(video_args)):
            if res is not None:
                results.append(res)
                
    out_df = pd.DataFrame(results)
    out_df.to_csv("svd_features_train.csv", index=False)
    print("Extraction complete. Saved to svd_features_train.csv.")
