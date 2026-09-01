import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def generate_dummy_calibration_data(num_samples=200):
    """
    Generates dummy probability scores from Module A (Visual) and Module B (Physio).
    target: 1 for 'zoned out', 0 for 'focused'.
    """
    np.random.seed(42)
    # Zoned out instances
    vis_prob_zoned = np.random.normal(0.2, 0.1, num_samples // 2)
    phys_prob_zoned = np.random.normal(0.3, 0.15, num_samples // 2)
    y_zoned = np.ones(num_samples // 2)
    
    # Focused instances
    vis_prob_focus = np.random.normal(0.8, 0.1, num_samples // 2)
    phys_prob_focus = np.random.normal(0.7, 0.15, num_samples // 2)
    y_focus = np.zeros(num_samples // 2)
    
    vis_prob = np.concatenate([vis_prob_zoned, vis_prob_focus])
    phys_prob = np.concatenate([phys_prob_zoned, phys_prob_focus])
    y = np.concatenate([y_zoned, y_focus])
    
    # Clip probabilities to [0, 1]
    vis_prob = np.clip(vis_prob, 0, 1)
    phys_prob = np.clip(phys_prob, 0, 1)
    
    X = np.vstack((vis_prob, phys_prob)).T
    return X, y

def calibrate_and_evaluate():
    print("Gathering Hardware Calibration Data (Simulated)...")
    X, y = generate_dummy_calibration_data(num_samples=500)
    
    # Split into train/test (80/20)
    split_idx = int(len(X) * 0.8)
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print("Training Random Forest Meta-Classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Class predictions for Classification Metrics
    y_pred_class = clf.predict(X_test)
    
    # Continuous probability predictions for Regression Metrics (R2, RMSE, MAE)
    # We use the probability of class 1 ('zoned out')
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
    print("\n--- Calibration Results ---")
    
    # Classification Metrics (from Proposal)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_class):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_class):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_class):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_class):.4f}")
    
    print("\n--- Continuous Meta-Classifier Metrics ---")
    # Continuous Metrics (Requested by User)
    print(f"R2 Score: {r2_score(y_test, y_pred_prob):.4f}")
    print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred_prob)):.4f}")
    print(f"MAE:      {mean_absolute_error(y_test, y_pred_prob):.4f}")
    
    print("\nCalibration Complete. Model ready for real-time fusion.")

if __name__ == "__main__":
    calibrate_and_evaluate()
