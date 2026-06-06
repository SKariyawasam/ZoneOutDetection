import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

class VisualInferenceModel:
    def __init__(self, checkpoint_dir="C:/Users/User/Desktop/FYP/DaiSEE Code/checkpoints/efficientnet_mixed_prec"):
        """Loads the EfficientNet-B0 pre-trained on DAiSEE."""
        self.img_size = 224
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Warning: Could not open webcam.")
            self.cap = None

        try:
            # Enable mixed precision to load the mixed_float16 checkpoint correctly
            set_global_policy('mixed_float16')
            
            # Match architecture built in train.py exactly
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
            
            efficientnet = tf.keras.applications.EfficientNetB0(weights=None, input_shape=(224, 224, 3), include_top=False)
            efficientnet.trainable = False
            model.add(efficientnet)
            
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(1024, activation='relu'))
            model.add(tf.keras.layers.Dense(256, activation='relu'))
            model.add(tf.keras.layers.Dense(4, activation='sigmoid', dtype='float32', name='prediction'))
            
            self.model = model
            
            # Use CheckpointManager to restore weights
            ckpt = tf.train.Checkpoint(net=self.model)
            manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
            if manager.latest_checkpoint:
                ckpt.restore(manager.latest_checkpoint).expect_partial()
                print(f"Successfully loaded visual model from {manager.latest_checkpoint}")
            else:
                print(f"Warning: Visual weights not found in {checkpoint_dir}. Using random weights.")
                
            # Initialize OpenCV Face Detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
        except Exception as e:
            print(f"Error initializing visual model: {e}")
            self.model = None

    def predict(self, features=None):
        """
        Captures a frame from the webcam and outputs an independent probability score for visual engagement.
        Returns: float (0.0 to 1.0) or -1.0 if no face is detected.
        """
        if self.model is None or self.cap is None:
            return 0.75 # Mock fallback
            
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            return 0.75
            
        try:
            # Face Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                return -1.0 # No face detected
                
            # Preprocess the frame
            # EfficientNetB0 expects RGB, OpenCV uses BGR
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Add batch dimension
            img_batch = np.expand_dims(img, axis=0)
            
            # Inference returns 4 values: Boredom, Engagement, Confusion, Frustration
            prediction = self.model.predict(img_batch, verbose=0)
            
            # We want Engagement, which is at index 1
            engagement_score = float(prediction[0][1])
            return engagement_score
            
        except Exception as e:
            print(f"Error during visual inference: {e}")
            return 0.75
            
    def release(self):
        if self.cap is not None:
            self.cap.release()
