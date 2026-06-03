import os
import cv2
import numpy as np
import tensorflow as tf

class VisualInferenceModel:
    def __init__(self, model_weights_path="c:/Users/User/Desktop/FYP/DaiSEE Code/checkpoints/model_weights.h5"):
        """Loads the EfficientNet-B0 pre-trained on DAiSEE."""
        self.img_size = 224
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Warning: Could not open webcam.")
            self.cap = None

        try:
            # Build the EfficientNetB0 base architecture
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_shape=(self.img_size, self.img_size, 3),
                pooling='avg'
            )
            
            # Recreate the classification head
            x = tf.keras.layers.Dense(512, activation='relu')(base_model.output)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
            
            self.model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
            
            if os.path.exists(model_weights_path):
                self.model.load_weights(model_weights_path)
                print(f"Successfully loaded visual model weights from {model_weights_path}")
            else:
                print(f"Warning: Visual weights not found at {model_weights_path}. Using random weights.")
                
        except Exception as e:
            print(f"Error initializing visual model: {e}")
            self.model = None

    def predict(self, features=None):
        """
        Captures a frame from the webcam and outputs an independent probability score for visual engagement.
        Returns: float (0.0 to 1.0)
        """
        if self.model is None or self.cap is None:
            return 0.75 # Mock fallback
            
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame from webcam")
            return 0.75
            
        try:
            # Preprocess the frame
            # EfficientNetB0 expects RGB, OpenCV uses BGR
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Add batch dimension
            img_batch = np.expand_dims(img, axis=0)
            
            # Inference
            prediction = self.model.predict(img_batch, verbose=0)
            return float(prediction[0][0])
            
        except Exception as e:
            print(f"Error during visual inference: {e}")
            return 0.75
            
    def release(self):
        if self.cap is not None:
            self.cap.release()
