# import numpy as np

class MetaClassifier:
    def __init__(self):
        """Late Fusion Meta-Classifier (Random Forest / Gradient Boosting placeholder)."""
        print("Initialized MetaClassifier.")

    def classify(self, visual_prob, physio_prob):
        """
        Takes probability outputs from Module A and Module B.
        Returns: binary classification ("focused" or "zoned out").
        """
        # Placeholder fusion logic
        combined_score = (visual_prob * 0.6) + (physio_prob * 0.4)
        if combined_score < 0.5:
            return "zoned out"
        return "focused"
