import cv2

def extract_features(frame):
    """
    Extracts facial expressions, eye gaze, and head posture.
    Returns lightweight features for the inference model.
    """
    # Placeholder for OpenCV / WebGazer equivalent feature extraction
    features = {
        "gaze": [0.0, 0.0],
        "head_pose": [0.0, 0.0, 0.0],
        "expression_landmarks": []
    }
    return features
