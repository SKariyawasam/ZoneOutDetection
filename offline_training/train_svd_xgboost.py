import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import pickle

def main():
    print("Loading extracted SVD features...")
    df = pd.read_csv("svd_features_train.csv")
    
    # Separate features and labels
    y = df['label']
    # Drop non-feature columns
    X = df.drop(columns=['label', 'video_path'])
    
    print(f"Dataset shape: {X.shape}")
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create XGBoost model
    # Using hist tree method which supports GPU acceleration if available
    clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device='cuda', # Use GPU
        eval_metric='auc',
        early_stopping_rounds=50,
        random_state=42
    )
    
    print("Training XGBoost Model...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Evaluate
    print("\n--- Evaluation on Validation Set ---")
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save the model
    model_path = "svd_xgboost_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")
    
    # Save the feature columns list so inference knows exactly what to pass
    cols_path = "svd_feature_cols.pkl"
    with open(cols_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    print(f"Feature columns saved to {cols_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
